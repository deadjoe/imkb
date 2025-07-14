"""
Test suite for CLI interface

Tests CLI commands, argument parsing, and integration with core functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from imkb.cli import cli, config, gen_playbook, get_rca


class TestCLICommands:
    """Test CLI command functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()

    def test_cli_group_help(self):
        """Test CLI group help command"""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "imkb - AI-powered incident knowledge base" in result.output
        assert "get-rca" in result.output
        assert "gen-playbook" in result.output
        assert "config" in result.output

    def test_cli_version(self):
        """Test CLI version display"""
        result = self.runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output or "version" in result.output.lower()


class TestGetRCACommand:
    """Test get-rca CLI command"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()

        # Create temporary event file
        self.event_data = {
            "id": "test-incident-123",
            "title": "MySQL Connection Pool Exhausted",
            "description": "Database connection pool has reached maximum capacity",
            "severity": "critical",
            "source": "monitoring",
            "metadata": {"database": "production_db", "max_connections": "150"},
        }

    def test_get_rca_help(self):
        """Test get-rca help command"""
        result = self.runner.invoke(get_rca, ["--help"])

        assert result.exit_code == 0
        assert "Analyze an incident event" in result.output
        assert "--event-file" in result.output
        assert "--namespace" in result.output

    def test_get_rca_missing_event_file(self):
        """Test get-rca without event file"""
        result = self.runner.invoke(get_rca, [])

        assert result.exit_code != 0
        # Click should complain about missing required option

    def test_get_rca_nonexistent_file(self):
        """Test get-rca with non-existent file"""
        result = self.runner.invoke(get_rca, ["--event-file", "nonexistent.json"])

        assert result.exit_code != 0
        # Click should validate file existence

    @patch("imkb.cli.get_rca_func")
    def test_get_rca_success(self, mock_get_rca):
        """Test successful get-rca execution"""
        # Mock the RCA function
        mock_get_rca.return_value = {
            "root_cause": "Connection pool configuration insufficient",
            "confidence": 0.85,
            "extractor": "mysqlkb",
            "status": "SUCCESS",
            "immediate_actions": ["Increase max_connections", "Monitor pool usage"],
            "references": [{"excerpt": "Pool sizing guide", "source": "mysql_kb"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.event_data, f)
            f.flush()

            try:
                result = self.runner.invoke(get_rca, ["--event-file", f.name])

                assert result.exit_code == 0
                assert "RCA Analysis Results" in result.output
                assert "Connection pool configuration insufficient" in result.output
                assert "Confidence: 0.85" in result.output
                assert "Extractor: mysqlkb" in result.output
                assert "Status: SUCCESS" in result.output
                assert "Immediate Actions:" in result.output
                assert "Increase max_connections" in result.output
                assert "Knowledge References: 1 items" in result.output

                # Verify the function was called with correct arguments
                mock_get_rca.assert_called_once()
                call_args = mock_get_rca.call_args
                assert call_args[0][0]["id"] == "test-incident-123"
                assert call_args[0][1] == "default"  # default namespace

            finally:
                Path(f.name).unlink()

    @patch("imkb.cli.get_rca_func")
    def test_get_rca_with_namespace(self, mock_get_rca):
        """Test get-rca with custom namespace"""
        mock_get_rca.return_value = {
            "root_cause": "Test cause",
            "confidence": 0.8,
            "extractor": "test",
            "status": "SUCCESS",
            "immediate_actions": [],
            "references": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.event_data, f)
            f.flush()

            try:
                result = self.runner.invoke(
                    get_rca, ["--event-file", f.name, "--namespace", "custom_tenant"]
                )

                assert result.exit_code == 0

                # Verify namespace was passed correctly
                call_args = mock_get_rca.call_args
                assert call_args[0][1] == "custom_tenant"

            finally:
                Path(f.name).unlink()

    @patch("imkb.cli.get_rca_func")
    def test_get_rca_error_handling(self, mock_get_rca):
        """Test get-rca error handling"""
        # Mock the function to raise an exception
        mock_get_rca.side_effect = Exception("RCA analysis failed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.event_data, f)
            f.flush()

            try:
                result = self.runner.invoke(get_rca, ["--event-file", f.name])

                assert (
                    result.exit_code == 0
                )  # CLI doesn't exit with error, just prints error
                assert "RCA analysis failed" in result.output
                assert "❌" in result.output

            finally:
                Path(f.name).unlink()

    def test_get_rca_invalid_json(self):
        """Test get-rca with invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            f.flush()

            try:
                result = self.runner.invoke(get_rca, ["--event-file", f.name])

                assert result.exit_code == 0  # CLI handles the error
                assert "❌" in result.output or "failed" in result.output.lower()

            finally:
                Path(f.name).unlink()


class TestGenPlaybookCommand:
    """Test gen-playbook CLI command"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()

        # Create sample RCA data
        self.rca_data = {
            "root_cause": "MySQL connection pool exhausted",
            "confidence": 0.85,
            "extractor": "mysqlkb",
            "references": [
                {
                    "excerpt": "Pool issue",
                    "source": "mysql_kb",
                    "confidence": 0.9,
                    "metadata": {},
                }
            ],
            "immediate_actions": ["Increase max_connections", "Monitor usage"],
            "preventive_measures": ["Add alerts", "Review config"],
        }

    def test_gen_playbook_help(self):
        """Test gen-playbook help command"""
        result = self.runner.invoke(gen_playbook, ["--help"])

        assert result.exit_code == 0
        assert "Generate remediation playbook" in result.output
        assert "--rca-file" in result.output
        assert "--namespace" in result.output

    def test_gen_playbook_missing_rca_file(self):
        """Test gen-playbook without RCA file"""
        result = self.runner.invoke(gen_playbook, [])

        assert result.exit_code != 0
        # Click should complain about missing required option

    @patch("imkb.cli.gen_playbook_func")
    def test_gen_playbook_success(self, mock_gen_playbook):
        """Test successful gen-playbook execution"""
        # Mock the playbook generation function
        mock_gen_playbook.return_value = {
            "actions": [
                "Increase MySQL max_connections parameter from 150 to 300",
                "Monitor connection pool usage for next 24 hours",
                "Review application connection handling",
            ],
            "playbook": "1. Execute SHOW PROCESSLIST 2. Increase max_connections 3. Monitor results",
            "priority": "high",
            "estimated_time": "30 minutes",
            "risk_level": "medium",
            "prerequisites": ["MySQL admin access", "Monitoring access"],
            "validation_steps": ["Check connection count", "Verify app functionality"],
            "rollback_plan": "Revert max_connections to 150 if issues arise",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.rca_data, f)
            f.flush()

            try:
                result = self.runner.invoke(gen_playbook, ["--rca-file", f.name])

                assert result.exit_code == 0
                assert "Remediation Playbook" in result.output
                assert "Priority: high" in result.output
                assert "Estimated Time: 30 minutes" in result.output
                assert "Risk Level: medium" in result.output
                assert "Actions (3):" in result.output
                assert "Increase MySQL max_connections" in result.output
                assert "Prerequisites:" in result.output
                assert "Validation Steps:" in result.output
                assert "Rollback Plan:" in result.output
                assert "Detailed Playbook:" in result.output

                # Verify the function was called correctly
                mock_gen_playbook.assert_called_once()
                call_args = mock_gen_playbook.call_args
                assert (
                    call_args[0][0]["root_cause"] == "MySQL connection pool exhausted"
                )
                assert call_args[0][1] == "default"  # default namespace

            finally:
                Path(f.name).unlink()

    @patch("imkb.cli.gen_playbook_func")
    def test_gen_playbook_with_namespace(self, mock_gen_playbook):
        """Test gen-playbook with custom namespace"""
        mock_gen_playbook.return_value = {
            "actions": ["Test action"],
            "playbook": "Test playbook",
            "priority": "medium",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.rca_data, f)
            f.flush()

            try:
                result = self.runner.invoke(
                    gen_playbook,
                    ["--rca-file", f.name, "--namespace", "production_tenant"],
                )

                assert result.exit_code == 0

                # Verify namespace was passed correctly
                call_args = mock_gen_playbook.call_args
                assert call_args[0][1] == "production_tenant"

            finally:
                Path(f.name).unlink()

    @patch("imkb.cli.gen_playbook_func")
    def test_gen_playbook_error_handling(self, mock_gen_playbook):
        """Test gen-playbook error handling"""
        # Mock the function to raise an exception
        mock_gen_playbook.side_effect = Exception("Playbook generation failed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.rca_data, f)
            f.flush()

            try:
                result = self.runner.invoke(gen_playbook, ["--rca-file", f.name])

                assert result.exit_code == 0  # CLI handles the error
                assert "Playbook generation failed" in result.output
                assert "❌" in result.output

            finally:
                Path(f.name).unlink()

    @patch("imkb.cli.gen_playbook_func")
    def test_gen_playbook_minimal_result(self, mock_gen_playbook):
        """Test gen-playbook with minimal result fields"""
        # Mock minimal response (some fields missing)
        mock_gen_playbook.return_value = {
            "actions": ["Basic action"],
            "playbook": "Basic playbook",
            "priority": "low",
            # Missing optional fields like estimated_time, prerequisites, etc.
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.rca_data, f)
            f.flush()

            try:
                result = self.runner.invoke(gen_playbook, ["--rca-file", f.name])

                assert result.exit_code == 0
                assert "Priority: low" in result.output
                assert "Estimated Time: Not specified" in result.output
                assert "Actions (1):" in result.output
                assert "Basic action" in result.output

            finally:
                Path(f.name).unlink()


class TestConfigCommand:
    """Test config CLI command"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()

    def test_config_help(self):
        """Test config help command"""
        result = self.runner.invoke(config, ["--help"])

        assert result.exit_code == 0
        assert "Manage imkb configuration" in result.output
        assert "--show" in result.output

    def test_config_no_options(self):
        """Test config command without options"""
        result = self.runner.invoke(config, [])

        assert result.exit_code == 0
        assert "Use --show to display current configuration" in result.output

    def test_config_show_placeholder(self):
        """Test config --show command (actual implementation)"""
        result = self.runner.invoke(config, ["--show"])

        assert result.exit_code == 0
        assert "Current imkb Configuration" in result.output


class TestCLIIntegration:
    """Integration tests for CLI commands"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()

    @patch("imkb.cli.get_rca_func")
    @patch("imkb.cli.gen_playbook_func")
    def test_rca_to_playbook_workflow(self, mock_gen_playbook, mock_get_rca):
        """Test complete workflow from RCA to playbook generation"""
        # Mock RCA response
        rca_result = {
            "root_cause": "Database connection timeout",
            "confidence": 0.9,
            "extractor": "mysqlkb",
            "status": "SUCCESS",
            "immediate_actions": ["Check connections", "Restart DB"],
            "references": [],
        }
        mock_get_rca.return_value = rca_result

        # Mock playbook response
        playbook_result = {
            "actions": ["Execute database restart", "Monitor connections"],
            "playbook": "Step-by-step database recovery procedure",
            "priority": "high",
            "risk_level": "medium",
        }
        mock_gen_playbook.return_value = playbook_result

        # Create test event file
        event_data = {
            "id": "workflow-test",
            "title": "Database Timeout",
            "description": "Connection timeouts detected",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as event_file:
            json.dump(event_data, event_file)
            event_file.flush()

            try:
                # Step 1: Generate RCA
                rca_cli_result = self.runner.invoke(
                    get_rca, ["--event-file", event_file.name]
                )
                assert rca_cli_result.exit_code == 0
                assert "Database connection timeout" in rca_cli_result.output

                # In real workflow, RCA would be saved to file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as rca_file:
                    # We use the mock rca_result dict, not the CLI output object
                    json.dump(rca_result, rca_file)
                    rca_file.flush()

                    try:
                        # Step 2: Generate playbook from RCA
                        playbook_result = self.runner.invoke(
                            gen_playbook, ["--rca-file", rca_file.name]
                        )
                        assert playbook_result.exit_code == 0
                        assert "Execute database restart" in playbook_result.output
                        assert "Priority: high" in playbook_result.output

                    finally:
                        Path(rca_file.name).unlink()

            finally:
                Path(event_file.name).unlink()

    def test_cli_with_real_example_files(self):
        """Test CLI with real example files from the project"""
        # Test with the actual example file if it exists
        example_file = Path("examples/mysql_event.json")

        if example_file.exists():
            result = self.runner.invoke(get_rca, ["--event-file", str(example_file)])

            # Should not crash, even if it uses mock LLM
            assert result.exit_code == 0
            assert "RCA Analysis Results" in result.output
        else:
            # Skip test if example file doesn't exist
            pytest.skip("Example file not found")

    def test_error_handling_with_corrupted_files(self):
        """Test CLI error handling with various corrupted input files"""
        test_cases = [
            ("", "empty file"),
            ("{", "incomplete JSON"),
            ('{"invalid": json}', "malformed JSON"),
            ('{"valid": "json", "but": "wrong_schema"}', "wrong schema"),
        ]

        for content, _description in test_cases:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(content)
                f.flush()

                try:
                    result = self.runner.invoke(get_rca, ["--event-file", f.name])

                    # CLI should handle errors gracefully, not crash
                    assert result.exit_code == 0
                    # Should show some error indication
                    assert "❌" in result.output or "failed" in result.output.lower()

                finally:
                    Path(f.name).unlink()


class TestCLIUsability:
    """Test CLI usability and user experience"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()

    def test_command_discovery(self):
        """Test that users can discover available commands"""
        result = self.runner.invoke(cli)

        # Should show help when no command is provided
        assert result.exit_code == 0 or "Usage:" in result.output

    def test_helpful_error_messages(self):
        """Test that error messages are helpful"""
        # Test missing required argument
        result = self.runner.invoke(get_rca)

        assert result.exit_code != 0
        # Should mention the missing argument
        assert "event-file" in result.output or "Missing option" in result.output

    def test_output_formatting(self):
        """Test that CLI output is well-formatted and readable"""
        # This would typically be tested with actual output inspection
        # For now, we ensure the key formatting elements are present

        with patch("imkb.cli.get_rca_func") as mock_get_rca:
            mock_get_rca.return_value = {
                "root_cause": "Test cause",
                "confidence": 0.8,
                "extractor": "test",
                "status": "SUCCESS",
                "immediate_actions": ["Action 1", "Action 2"],
                "references": [{"test": "reference"}],
            }

            event_data = {"id": "test", "title": "Test", "description": "Test"}

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(event_data, f)
                f.flush()

                try:
                    result = self.runner.invoke(get_rca, ["--event-file", f.name])

                    # Check for clear section headers
                    assert "🔍" in result.output  # emoji for visual appeal
                    assert "=" in result.output  # section separators
                    assert "1." in result.output  # numbered lists

                finally:
                    Path(f.name).unlink()
