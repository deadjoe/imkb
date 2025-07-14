# imkb 库 · 详细设计文档

## 项目概述

**imkb** 是一套独立的 Python SDK，用于在运维场景中把"故障/告警事件"转化为 **AI 可推理的上下文**，并帮助本地或远端大语言模型（LLM）产出**根因分析 (RCA)** 与**修复建议**。

### 核心定位
- 聚焦*知识检索*与*推理编排*
- 不依赖于任何特定的告警平台或执行框架
- 可被 CLI 工具、Bot、Web 服务等多种上层应用直接调用

## 设计目标

| ID | 目标 | 量化指标 |
|---|------|----------|
|G1|提供统一**事件→RCA**与**RCA→Action** API|`get_rca()` P95 < 800ms|
|G2|支持**多知识库/多LLM/多存储后端**即插即用|新增 Extractor/LLM 无需改核心|
|G3|内置**召回(Recall) + 图关联 + Token 削减**策略|召回 Recall@10 ≥ 0.9；Prompt Token↓ 60%|
|G4|生产可观测：OpenTelemetry Trace、Prometheus 指标|四大 span (`recall/kb_query/llm_infer/save`)|

## 核心概念

| 概念 | 说明 |
|------|------|
|**Extractor**|面向单一知识源（RHOKP、MySQLKB…）的检索插件|
|**Recall**|通过 Mem0 的向量+图混合算法从记忆层捞出候选片段|
|**LLM Client**|封装本地/远端 LLM（DeepSeek、Llama.cpp、OpenAI…）|
|**RCA Pipeline**|高阶流程：事件 → 召回 → LLM → `RCAResult`|
|**Action Pipeline**|根因 → 历史 Playbook 召回 → 新修复脚本|
|**Adapter**|对接外部后端，如 Mem0、Qdrant、Neo4j、Solr、Playwright|
|**Namespace**|`<env>-<org_id>` 分库隔离|

## 系统架构

```mermaid
graph TD
    subgraph "imkb SDK"
        EVENT["事件 JSON"]
        PIPE["RCA Pipeline"]
        REC["Recall Layer"]
        MEM["Mem0 Hybrid Store"]
        VEC["Vector Search"]
        GRAPH["Graph Traversal"]
        KB["Knowledge Base Extractors"]
        LLM["LLM Client Router"]
        RES["RCAResult"]
        ACTION["Action Pipeline"]
        PLAYBOOK["Playbook Result"]

        EVENT --> PIPE
        PIPE --> REC
        REC --> MEM
        MEM --> VEC
        MEM --> GRAPH
        REC --> KB
        PIPE --> LLM
        LLM --> PIPE
        PIPE --> RES
        RES --> ACTION
        ACTION --> PLAYBOOK
    end

    subgraph "External Systems"
        SOLR["Solr KB"]
        MYSQL["MySQL KB"]
        PLAYWRIGHT["Web Scraping"]
        LOCAL["Local llama.cpp"]
        CLOUD["Cloud APIs"]
        QDRANT["Qdrant Vector Store"]
        NEO4J["Neo4j Graph Store"]

        KB --> SOLR
        KB --> MYSQL
        KB --> PLAYWRIGHT

        LLM --> LOCAL
        LLM --> CLOUD

        MEM --> QDRANT
        MEM --> NEO4J
    end

```

## 延迟预算分配

| 层级 | 关键操作 | 目标 P95 | 方案要点 |
|------|----------|----------|----------|
|Mem0 向量召回|vector.search + Top-K 过滤|≤ 120ms|HNSW + quantization；batch 查询；缓存最近 1min hit|
|图关联 (Neo4j)|MATCH … 1–2 跳|≤ 80ms|预建索引；限定最多两跳；同向遍历|
|KB 外呼 (Solr)|JSON API|≤ 150ms|只在向量 miss 时调用；结果写 Mem0，1h 内命中不再外呼|
|Playwright 抓取|浏览器抓取|≤ 300ms|仅作为最后兜底；触发熔断阈值时降级|
|Prompt 生成 & LLM 推理|Jinja2 渲染 + LLM 调用|本地 LLM ≤ 300ms<br/>云 API ≤ 500ms|KV-cache 复用；Token ≤ 1k；并发令牌桶|
|其它 (序列化、链路)|JSON encode / OTLP span|≤ 50ms|uvloop / orjson|

**预算汇总**：
- Cache 命中路径：≈ 470ms
- 冷路径（含 Solr）：≈ 700ms  
- 极端冷路径（Playwright）：≈ 950ms → 触发降级

## 包结构设计

```
imkb/
├── __init__.py               # 导出顶级 API
├── adapters/                 # 第三方后端适配层
│   ├── __init__.py
│   ├── mem0.py              # Mem0 SDK 包装
│   ├── qdrant.py
│   ├── neo4j.py
│   ├── solr.py
│   └── playwright.py
├── extractors/              # 知识源插件目录
│   ├── __init__.py          # 注册表
│   ├── base.py              # BaseExtractor Protocol
│   ├── rhokp.py             # Red Hat 知识库
│   ├── mysqlkb.py           # MySQL 故障库
│   └── postgreskb.py        # PostgreSQL 知识库
├── prompts/                 # Jinja2 模板 & metadata
│   ├── mysql_rca/
│   │   ├── v1/
│   │   │   ├── template.jinja2
│   │   │   └── meta.yaml
│   │   └── v2/
│   └── postgres_rca/
├── llm_client.py            # Async Router to local/remote LLM
├── recall.py                # 核心召回策略（Mem0 hybrid）
├── rca_pipeline.py          # 高阶 API：get_rca()
├── action_pipeline.py       # 高阶 API：gen_playbook()
├── config.py                # pydantic Settings
├── telemetry.py             # OpenTelemetry & metrics
└── cli.py                   # `python -m imkb ...`
```

## 核心数据模型

### Event Schema

```python
class Event(TypedDict):
    id: str                  # UUID
    signature: str           # hash-or-error_code，用于去重与向量化
    context_hash: str        # sha256(labels+message前64chars)
    timestamp: str           # ISO8601
    severity: Literal["P0", "P1", "P2", "P3"]
    source: str              # "prometheus", "alertmanager", etc.
    labels: dict[str, str]   # {"cluster": "prod-eu1", ...}
    message: str             # 限制 256 chars
    raw: dict                # 原始告警全文
    embedding_version: str   # "v1.0" 支持向量模型升级
```

### Knowledge Base Item

```python
class KBItem(TypedDict):
    doc_id: str
    excerpt: str             # 文档片段
    score: float             # 0.0-1.0 相似度分数
    metadata: dict           # 来源、时间戳等元信息
```

### RCA Result

```python
class RCAResult(BaseModel):
    root_cause: str          # 根因分析文本
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0-1.0 置信度
    extractor: str           # 使用的提取器名称
    references: list[KBItem] = Field(default_factory=list) # 引用的知识片段
    status: str = "SUCCESS"  # SUCCESS|PARSE_ERROR|NO_CONTEXT|LLM_ERROR
    contributing_factors: list[str] = Field(default_factory=list)     # 贡献因素
    evidence: list[str] = Field(default_factory=list)                 # 证据链
    immediate_actions: list[str] = Field(default_factory=list)        # 立即行动
    preventive_measures: list[str] = Field(default_factory=list)      # 预防措施
    additional_investigation: list[str] = Field(default_factory=list) # 需要额外调查的点
    confidence_reasoning: str = ""           # 置信度推理
    knowledge_gaps: list[str] = Field(default_factory=list)           # 知识盲点
    metadata: dict[str, Any] = Field(default_factory=dict)            # 元数据
```

## Extractor 架构

### BaseExtractor Protocol

```python
class BaseExtractor(Protocol):
    name: str
    prompt_template: str     # 指向 prompts/ 目录的模板路径
    
    async def match(self, event: Event) -> bool:
        """判断插件是否适用（基于标签/关键字/资源类型）"""
        ...
    
    async def recall(self, event: Event, k: int = 10) -> list[KBItem]:
        """仅负责检索数据片段（Mem0＋KB），不涉 LLM"""
        ...
    
    def get_prompt_context(self, event: Event, snippets: list[KBItem]) -> dict:
        """定制化 prompt 上下文构建"""
        ...
```

### 插件示例：MySQL KB Extractor

```python
class MySQLKBExtractor:
    name = "mysqlkb"
    prompt_template = "mysql_rca:v1"
    
    async def match(self, event: Event) -> bool:
        return "mysql" in event["labels"].get("service", "").lower()
    
    async def recall(self, event: Event, k: int = 10) -> list[KBItem]:
        # 1. Mem0 向量+图混合召回
        mem0_results = await self.mem0_client.search(
            query=event["signature"],
            user_id=f"{namespace}-mysql"
        )
        
        # 2. 如果结果不足，查询外部 MySQL KB
        if len(mem0_results) < 3:
            sql_results = await self.query_mysql_kb(event)
            # 写回 Mem0
            await self.mem0_client.add_batch(sql_results)
            
        return self.convert_to_kbitems(mem0_results[:k])
```

## 召回融合策略

### Mem0 混合召回流程

1. **向量先行**：`qdrant.search(signature)` Top-K=8
2. **图补全**：对命中最高的 fact 取 `(fact)-[:CAUSES]->(insight)` 1跳
3. **片段去噪**：动态阈值过滤
   - 如果 Top-3 平均 score > 0.8，阈值 = 0.75
   - 否则降为 0.6，确保最少 3 个片段
4. **不足补充**：< 3 片段时 → KB 外呼
5. **写回更新**：KB 片段入向量库并建图边 `(fact)-[:EVIDENCE]->(kb)`

## 多租户隔离

### 安全隔离方案

| 层 | 隔离键 | 实现方式 |
|---|--------|----------|
|配置|`namespace = f"{env}-{org_id}"`|注入到 ContextVar，全链路透传|
|Mem0-Qdrant|`collection = "vec_" + namespace`|每租户独立 collection|
|Mem0-Graph (Neo4j)|使用 database 级隔离|每租户独立 database，避免 Cypher 注入|
|Solr/SQL KB|core/schema 每租户前缀|Adapter 在查询时拼接租户标识|
|LLM Router|路由规则按租户映射|YAML：`llm.routers.<namespace>.provider`|

### 查询审计

所有跨租户访问尝试记录到审计日志，包含：
- 访问时间、用户ID、目标 namespace
- 查询内容 hash
- 是否成功及失败原因

## 配置系统

### 主配置文件 (imkb.yml)

```yaml
llm:
  default: "deepseek_local"
  routers:
    deepseek_local:
      provider: "llama_cpp"
      model: "deepseek-33b.awq"
      gpu_layers: 20
      max_tokens: 1024
    openai_cloud:
      provider: "openai"
      model: "gpt-4o-mini"
      api_key: "${OPENAI_API_KEY}"

mem0:
  vector_store:
    provider: "qdrant"
    host: "localhost"
    port: 6333
    embedding_model_dims: 1536
  graph_store:
    provider: "neo4j"
    url: "neo4j://localhost:7687"
    username: "neo4j"
    password: "${NEO4J_PASSWORD}"

extractors:
  enabled:
    - rhokp
    - mysqlkb
  rhokp:
    solr_url: "https://rhokp.local/solr"
    timeout: 5.0
  mysqlkb:
    connection_string: "mysql://user:pass@localhost/kb"
    cache_ttl: 3600

features:
  mem0_graph: true
  solr_kb: true
  playwright_kb: false
  local_llm: true
  cloud_llm: true

telemetry:
  otlp_endpoint: "http://localhost:4317"
  enable_metrics: true
  enable_tracing: true
```

## 错误处理与降级策略

| 故障类型 | 处理策略 | 影响 |
|----------|----------|------|
|LLM 超时/5xx|返回 `confidence=0` + `status="LLM_UNAVAILABLE"`|保持链路存档|
|Mem0 写失败|重试 3 次 → fallback to Redis buffer；异步补写|不阻塞 RCA 输出|
|KB 外呼 429|触发 `kb_status=degraded`，30s 静默；仅靠向量召回|P95 延迟稳定|
|召回完全为空|返回 `status="NO_CONTEXT"`|值守人员介入|
|延迟超过阈值|熔断机制，跳过非核心步骤|保证基本可用性|

## Prompt 管理与版本化

### 模板结构

```
prompts/mysql_rca/v1/
├── template.jinja2
└── meta.yaml
```

**template.jinja2**:
```jinja2
You are an expert MySQL DBA analyzing a production incident.

## Event Details
- ID: {{ event.id }}
- Severity: {{ event.severity }}
- Message: {{ event.message }}
- Labels: {{ event.labels | tojson }}

## Relevant Knowledge
{% for snippet in snippets %}
- **{{ snippet.doc_id }}** (score: {{ snippet.score }}): {{ snippet.excerpt }}
{% endfor %}

## Task
Provide root cause analysis in JSON format:
{
  "root_cause": "detailed explanation",
  "confidence": 0.85,
  "next_steps": ["step1", "step2"]
}
```

**meta.yaml**:
```yaml
version: "v1"
min_llm_context_window: 4096
inputs:
  - event
  - snippets
output_schema: "RCAResult"
description: "MySQL incident root cause analysis"
```

### 版本升级机制

```python
# 热切换到新版本
await prompt_manager.upgrade("mysql_rca", "v2")

# 回滚到上一版本
await prompt_manager.rollback("mysql_rca")
```

## MVP 实现计划

### Phase 1: 核心召回 + 推理
- [x] Mem0 向量+图混合召回
- [x] 单一 LLM 客户端（本地 llama.cpp）
- [x] MySQL KB Extractor
- [x] 基础错误处理

### Phase 2: 生产化
- [ ] 多租户隔离
- [ ] OpenTelemetry 可观测性
- [ ] 多 LLM 路由
- [ ] Action Pipeline

### Phase 3: 扩展
- [ ] 更多 Extractor 插件
- [ ] Web UI
- [ ] 性能优化

## 依赖管理

### pyproject.toml 配置

```toml
[project.optional-dependencies]
mem0 = ["mem0-ai>=1.0", "qdrant-client>=1.7"]
graph = ["neo4j>=5.0"]
kb = ["solr>=2.0", "playwright>=1.40"]
llm-gpu = ["llama-cpp-python", "torch"]
all = ["imkb[mem0,graph,kb,llm-gpu]"]
```

### 安装示例

```bash
# MVP 最小安装
pip install imkb[mem0,graph]

# 完整功能
pip install imkb[all]
```

## 可观测性

### OpenTelemetry Spans

- `imkb.get_rca` (root span)
  - `recall.mem0_search`
  - `recall.kb_query` 
  - `llm.infer`
  - `mem0.save_results`

### Prometheus 指标

- `imkb_rca_duration_seconds`
- `imkb_recall_hit_rate`
- `imkb_llm_token_usage`
- `imkb_error_count_total`

## 总结

imkb 设计为一个高性能、可扩展的 AI 运维知识检索与推理 SDK。通过 Mem0 的向量+图混合架构，实现了高效的知识召回和关系推理。模块化的 Extractor 系统支持多种知识源的无缝集成，而统一的 LLM 客户端提供了灵活的模型选择。

设计重点关注生产环境的性能要求（P95 < 800ms）、多租户安全隔离，以及完整的可观测性支持，为企业级运维 AI 应用提供了坚实的技术基础。