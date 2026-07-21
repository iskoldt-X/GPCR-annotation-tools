"""Tests for the audit trail logger."""

import json


class TestLogAuditTrail:
    def test_creates_entry(self, configure_paths):
        """An audit trail entry should be appended to the JSONL file."""
        _, output_dir = configure_paths

        from gpcr_tools.csv_generator.audit import log_audit_trail

        log_audit_trail("TEST1", "receptor_info.chain_id", "accept", "A", "A")

        audit_file = output_dir / "audit" / "audit_trail.jsonl"
        assert audit_file.exists()

        with open(audit_file) as f:
            lines = f.readlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["pdb_id"] == "TEST1"
        assert entry["field_path"] == "receptor_info.chain_id"
        assert entry["action"] == "accept"

    def test_multiple_entries(self, configure_paths):
        """Multiple audit entries should be appended sequentially."""
        _, output_dir = configure_paths

        from gpcr_tools.csv_generator.audit import log_audit_trail

        log_audit_trail("TEST1", "field_a", "accept", "x", "x")
        log_audit_trail("TEST1", "field_b", "edit", "old", "new")
        log_audit_trail("TEST2", "field_c", "skip", None, None)

        audit_file = output_dir / "audit" / "audit_trail.jsonl"
        with open(audit_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

        for line in lines:
            entry = json.loads(line)
            assert "pdb_id" in entry
            assert "timestamp" in entry

    def test_soft_fails_when_audit_dir_cannot_be_created(self, tmp_path, monkeypatch):
        """A filesystem failure creating the audit directory must never crash
        curate: log_audit_trail should soft-fail (not raise) even when the
        audit directory cannot be created.

        Here a plain file is placed where the output directory is expected, so
        the audit directory resolves under a non-directory and mkdir raises.
        """
        from gpcr_tools.config import reset_config

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))

        # A regular file where the output directory should be: the audit dir
        # then resolves to <file>/audit, whose mkdir cannot succeed.
        output_as_file = tmp_path / "output_is_a_file"
        output_as_file.write_text("not a directory")
        monkeypatch.setenv("GPCR_OUTPUT_PATH", str(output_as_file))

        reset_config()
        try:
            from gpcr_tools.csv_generator.audit import log_audit_trail

            # Must not raise despite the un-creatable audit directory.
            log_audit_trail("TEST1", "field_a", "accept", "x", "x")
        finally:
            reset_config()
