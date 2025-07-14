"""
Command-line interface for imkb

Provides CLI commands for:
- Analyzing events: imkb get-rca --event-file event.json
- Generating playbooks: imkb gen-playbook --rca-file rca.json
- Managing configuration: imkb config --show
"""

import asyncio
import json

import click
import yaml

from . import __version__
from . import gen_playbook as gen_playbook_func
from . import get_rca as get_rca_func
from .config import get_config


@click.group()
@click.version_option(version=__version__, prog_name="imkb")
def cli():
    """imkb - AI-powered incident knowledge base and RCA analysis"""


@cli.command()
@click.option(
    "--event-file",
    type=click.Path(exists=True),
    required=True,
    help="JSON file containing event data",
)
@click.option("--namespace", default="default", help="Tenant namespace for isolation")
def get_rca(event_file: str, namespace: str):
    """Analyze an incident event and generate root cause analysis"""
    try:
        # Load event data
        with open(event_file) as f:
            event_data = json.load(f)

        # Run RCA analysis
        result = asyncio.run(get_rca_func(event_data, namespace))

        # Output result
        click.echo(f"🔍 RCA Analysis Results for {event_file}")
        click.echo("=" * 50)
        click.echo(f"Root Cause: {result['root_cause']}")
        click.echo(f"Confidence: {result['confidence']}")
        click.echo(f"Extractor: {result['extractor']}")
        # 输出状态，并在非 SUCCESS 情况下添加失败标记，方便测试捕获
        if result["status"] != "SUCCESS":
            click.echo(f"❌ Status: {result['status']}")
        else:
            click.echo(f"Status: {result['status']}")

        if result.get("immediate_actions"):
            click.echo("\nImmediate Actions:")
            for i, action in enumerate(result["immediate_actions"], 1):
                click.echo(f"  {i}. {action}")

        if result.get("references"):
            click.echo(f"\nKnowledge References: {len(result['references'])} items")

    except Exception as e:
        click.echo(f"❌ RCA analysis failed: {e}", err=True)


@cli.command()
@click.option(
    "--rca-file",
    type=click.Path(exists=True),
    required=True,
    help="JSON file containing RCA results",
)
@click.option("--namespace", default="default", help="Tenant namespace for isolation")
def gen_playbook(rca_file: str, namespace: str):
    """Generate remediation playbook from RCA analysis"""
    try:
        # Load RCA data
        with open(rca_file) as f:
            rca_data = json.load(f)

        # Generate playbook
        result = asyncio.run(gen_playbook_func(rca_data, namespace))

        # Output result
        click.echo(f"📋 Remediation Playbook for {rca_file}")
        click.echo("=" * 50)

        click.echo(f"Priority: {result.get('priority', 'Not specified')}")
        click.echo(f"Estimated Time: {result.get('estimated_time', 'Not specified')}")
        click.echo(f"Risk Level: {result.get('risk_level', 'Not specified')}")

        click.echo(f"\n🎯 Actions ({len(result.get('actions', []))}):")
        for i, action in enumerate(result.get("actions", []), 1):
            click.echo(f"  {i}. {action}")

        if result.get("prerequisites"):
            click.echo("\n📋 Prerequisites:")
            for prereq in result["prerequisites"]:
                click.echo(f"  - {prereq}")

        if result.get("validation_steps"):
            click.echo("\n✅ Validation Steps:")
            for step in result["validation_steps"]:
                click.echo(f"  - {step}")

        if result.get("rollback_plan"):
            click.echo("\n🔄 Rollback Plan:")
            click.echo(f"  {result['rollback_plan']}")

        click.echo("\n📖 Detailed Playbook:")
        click.echo(f"  {result['playbook']}")

    except Exception as e:
        click.echo(f"❌ Playbook generation failed: {e}", err=True)


@cli.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--format", type=click.Choice(["yaml", "json"]), default="yaml", help="Output format")
def config(show: bool, format: str):
    """Manage imkb configuration"""
    if show:
        try:
            config_obj = get_config()
            config_dict = config_obj.model_dump()

            click.echo("🔧 Current imkb Configuration")
            click.echo("=" * 40)

            if format == "yaml":
                click.echo(yaml.dump(config_dict, default_flow_style=False, indent=2))
            elif format == "json":
                click.echo(json.dumps(config_dict, indent=2))

        except Exception as e:
            click.echo(f"❌ Failed to load configuration: {e}", err=True)
    else:
        click.echo("Use --show to display current configuration")
        click.echo("Available options:")
        click.echo("  --show          Show current configuration")
        click.echo("  --format yaml   Output in YAML format (default)")
        click.echo("  --format json   Output in JSON format")


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
