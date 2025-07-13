
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

from . import __version__
from . import get_rca as get_rca_func
from . import gen_playbook as gen_playbook_func


@click.group()
@click.version_option(version=__version__, prog_name="imkb")
def cli():
    """imkb - AI-powered incident knowledge base and RCA analysis"""
    pass


@cli.command()
@click.option("--event-file", type=click.Path(exists=True), required=True, help="JSON file containing event data")
@click.option("--namespace", default="default", help="Tenant namespace for isolation")
def get_rca(event_file: str, namespace: str):
    """Analyze an incident event and generate root cause analysis"""
    try:
        # Load event data
        with open(event_file, 'r') as f:
            event_data = json.load(f)
        
        # Run RCA analysis
        result = asyncio.run(get_rca_func(event_data, namespace))
        
        # Output result
        click.echo(f"🔍 RCA Analysis Results for {event_file}")
        click.echo("=" * 50)
        click.echo(f"Root Cause: {result['root_cause']}")
        click.echo(f"Confidence: {result['confidence']}")
        click.echo(f"Extractor: {result['extractor']}")
        click.echo(f"Status: {result['status']}")
        
        if result.get('immediate_actions'):
            click.echo(f"\nImmediate Actions:")
            for i, action in enumerate(result['immediate_actions'], 1):
                click.echo(f"  {i}. {action}")
        
        if result.get('references'):
            click.echo(f"\nKnowledge References: {len(result['references'])} items")
        
    except Exception as e:
        click.echo(f"❌ RCA analysis failed: {e}", err=True)


@cli.command()
@click.option("--rca-file", type=click.Path(exists=True), required=True, help="JSON file containing RCA results")
@click.option("--namespace", default="default", help="Tenant namespace for isolation")
def gen_playbook(rca_file: str, namespace: str):
    """Generate remediation playbook from RCA analysis"""
    try:
        # Load RCA data
        with open(rca_file, 'r') as f:
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
        for i, action in enumerate(result.get('actions', []), 1):
            click.echo(f"  {i}. {action}")
        
        if result.get('prerequisites'):
            click.echo(f"\n📋 Prerequisites:")
            for prereq in result['prerequisites']:
                click.echo(f"  - {prereq}")
        
        if result.get('validation_steps'):
            click.echo(f"\n✅ Validation Steps:")
            for step in result['validation_steps']:
                click.echo(f"  - {step}")
        
        if result.get('rollback_plan'):
            click.echo(f"\n🔄 Rollback Plan:")
            click.echo(f"  {result['rollback_plan']}")
        
        click.echo(f"\n📖 Detailed Playbook:")
        click.echo(f"  {result['playbook']}")
        
    except Exception as e:
        click.echo(f"❌ Playbook generation failed: {e}", err=True)



@cli.command()
@click.option("--show", is_flag=True, help="Show current configuration")
def config(show: bool):
    """Manage imkb configuration"""
    if show:
        click.echo("Configuration management coming soon!")
    else:
        click.echo("Use --show to display current configuration")


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()