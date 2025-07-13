# imkb 项目开发记忆

## 项目状态 (2025-07-13)
- **仓库地址**: https://github.com/deadjoe/imkb
- **开发状态**: MVP 已完成并开源
- **最新提交**: ed70e3e - Initial release with core functionality
- **分支**: main (已推送到远程)

## 项目简介
imkb 是一个 Python SDK，用于运维场景中将故障/告警事件转化为 AI 可推理的上下文，并帮助 LLM 产出根因分析(RCA)与修复建议。

## 核心特性
- **混合召回**：基于 Mem0 的向量+图混合存储，实现高质量知识检索
- **插拔式架构**：Extractor 系统支持多种知识源（MySQL KB、Solr、Web抓取等）
- **多模型支持**：统一 LLM 客户端支持本地（llama.cpp）和云端（OpenAI）模型
- **生产就绪**：P95 < 800ms 性能目标，完整的可观测性和多租户隔离

## 技术栈
- **包管理**：uv
- **核心依赖**：
  - mem0-ai: 向量+图混合存储
  - qdrant-client: 向量数据库客户端  
  - neo4j: 图数据库
  - pydantic: 配置管理
  - jinja2: Prompt 模板
  - opentelemetry: 可观测性
- **可选依赖**：
  - llama-cpp-python: 本地 LLM
  - playwright: Web 抓取
  - solr: 搜索引擎

## 项目结构
```
imkb/
├── __init__.py               # 顶级 API
├── adapters/                 # 外部系统适配器
├── extractors/              # 知识源插件
├── prompts/                 # Jinja2 模板
├── llm_client.py            # LLM 路由客户端
├── recall.py                # Mem0 混合召回
├── rca_pipeline.py          # 主要 API: get_rca()
├── action_pipeline.py       # 次要 API: gen_playbook()
├── config.py                # 配置管理
├── telemetry.py             # 可观测性
└── cli.py                   # 命令行接口
```

## 核心工作流
1. 事件输入 → ExtractorRouter.match() 选择合适的插件
2. 插件.recall() → Mem0 向量+图混合检索 + 外部 KB 查询
3. RcaPipeline → Jinja2 渲染 Prompt + LLM 推理
4. 返回 RCAResult（根因、置信度、引用片段）

## 关键设计决策

### Mem0 混合召回策略
- 向量搜索 Top-K=8，图关系 1-2 跳遍历
- 动态阈值：Top-3 平均 score > 0.8 时阈值=0.75，否则=0.6
- 结果不足时触发外部 KB 查询并写回 Mem0

### 延迟预算分配（P95目标）
- Mem0 向量召回: ≤120ms
- Neo4j 图查询: ≤80ms  
- Solr KB 查询: ≤150ms
- LLM 推理: ≤300ms（本地）/≤500ms（云端）
- 其他开销: ≤50ms

### 多租户隔离
- namespace = f"{env}-{org_id}"
- Qdrant: 每租户独立 collection
- Neo4j: 每租户独立 database（安全考虑）
- 全链路 ContextVar 传递

### 错误处理与降级
- LLM 不可用: 返回 confidence=0，status="LLM_UNAVAILABLE"
- 召回为空: status="NO_CONTEXT"，人工介入
- KB 外呼限流: 30s 静默，仅依赖向量召回
- 延迟超阈值: 熔断非核心步骤

## MVP 实现状态 ✅
**Phase 1 已完成**:
- ✅ 配置系统 (config.py) - 支持 YAML 配置和环境变量
- ✅ 事件和知识项结构 (extractors/base.py)
- ✅ 提取器系统 (extractors/) - MySQL和测试提取器
- ✅ LLM 客户端 (llm_client.py) - 支持 OpenAI 和本地模型
- ✅ RCA 管道 (rca_pipeline.py) - 核心分析逻辑
- ✅ Action 管道 (action_pipeline.py) - 修复建议生成
- ✅ Mem0 适配器 (adapters/mem0.py) - 向量+图存储
- ✅ CLI 界面 (cli.py) - 命令行工具
- ✅ 可观测性系统 (observability/) - 指标和追踪
- ✅ 缓存和并发控制 (cache/, concurrency/)
- ✅ 完整测试覆盖率 (tests/)

**已实现的核心模块**:
- 51 个源文件
- 12,710+ 行代码
- 完整的 MVP 功能实现

## 开发环境
- Python 3.9+
- uv 包管理器
- 本地服务：Qdrant (6333), Neo4j (7687)

## 常用命令
```bash
# 初始化项目
uv init

# 安装依赖
uv add "mem0-ai>=1.0" "qdrant-client>=1.7" "neo4j>=5.0"

# 运行测试
uv run pytest

# 启动CLI
uv run python -m imkb get-rca --event-file event.json
```

## 配置文件
主配置: `imkb.yml`
- LLM 路由配置
- Mem0 向量+图存储配置  
- Extractor 插件配置
- 特性开关
- 可观测性配置

## 重要提醒
- 始终使用 async/await 模式
- 所有外部调用都要有超时和重试
- 遵循 OpenTelemetry 规范记录 span
- 配置使用 pydantic Settings，支持环境变量覆盖
- Prompt 模板版本化，支持热更新

## 开发工作流程
```bash
# 克隆项目
git clone https://github.com/deadjoe/imkb.git
cd imkb

# 安装依赖
uv sync

# 运行测试
uv run pytest

# 启动开发环境
uv run python test_mvp.py

# 运行 CLI
uv run imkb get-rca --event-file examples/event.example.json
```

## 下次协作信息
- 项目已完成 MVP 开发并成功开源
- GitHub 仓库已设置，包含完整源代码和文档
- 安全审核已完成，无敏感信息泄露
- 所有核心功能模块已实现并通过测试
- 可以开始后续功能开发或部署准备工作