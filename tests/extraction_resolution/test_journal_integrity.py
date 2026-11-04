"""Hash-chain and torn-tail tests for extraction resolution decisions."""

from __future__ import annotations

import json

import pytest

from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
from slr_pipeline.extraction.resolution_journal import (
    ResolutionDecisionIntent,
    append_decision_entry,
    decision_journal_path,
    read_decision_journal,
    repair_torn_journal_tail,
)


def _intent(run_id: str, *, field_id: str = "field-a", resolved_at: str = "2026-09-18T12:00:00Z"):
    return ResolutionDecisionIntent(
        run_id=run_id, record_id="record-a", field_id=field_id, outcome="not_reported",
        reviewer="reviewer", note="checked source", resolved_at=resolved_at,
        decision_origin="human",
    )


def test_torn_final_line_is_physically_removed_before_future_append(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="run-a")
    intent = _intent(store.run_id)
    append_decision_entry(store, intent, status="prepared")
    path = decision_journal_path(store)
    with open(path, "ab") as handle:
        handle.write(b'{"status":"appl')

    repaired = repair_torn_journal_tail(store)
    assert [entry.status for entry in repaired] == ["prepared"]
    assert path.read_bytes().endswith(b"\n")
    assert b'"status":"appl' not in path.read_bytes()
    append_decision_entry(store, intent, status="failed", reason="interrupted")
    assert [entry.status for entry in read_decision_journal(store)] == ["prepared", "failed"]


def test_syntactically_valid_tampered_final_line_is_not_repaired_away(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="run-a")
    intent = _intent(store.run_id)
    append_decision_entry(store, intent, status="prepared")
    path = decision_journal_path(store)
    line = json.loads(path.read_text(encoding="utf-8"))
    line["reviewer"] = "attacker"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="entry_hash"):
        repair_torn_journal_tail(store)
    assert path.read_bytes() == before


def test_complete_but_unterminated_final_line_is_kept_and_newline_repaired(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="run-a")
    intent = _intent(store.run_id)
    append_decision_entry(store, intent, status="prepared")
    path = decision_journal_path(store)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="unterminated final line"):
        read_decision_journal(store)
    repaired = repair_torn_journal_tail(store)
    assert [entry.status for entry in repaired] == ["prepared"]
    assert path.read_bytes().endswith(b"\n")


def test_deleted_prior_entry_breaks_previous_hash_chain(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="run-a")
    first = _intent(store.run_id, field_id="field-a", resolved_at="2026-09-18T12:00:00Z")
    second = _intent(store.run_id, field_id="field-b", resolved_at="2026-09-18T12:01:00Z")
    append_decision_entry(store, first, status="prepared")
    append_decision_entry(store, first, status="applied")
    append_decision_entry(store, second, status="prepared")
    path = decision_journal_path(store)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[2])
    with pytest.raises(ValueError, match="hash chain is broken"):
        read_decision_journal(store)


def test_reordered_entries_break_previous_hash_chain(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="run-a")
    intent = _intent(store.run_id)
    append_decision_entry(store, intent, status="prepared")
    append_decision_entry(store, intent, status="applied")
    path = decision_journal_path(store)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[1] + lines[0])
    with pytest.raises(ValueError, match="hash chain is broken"):
        read_decision_journal(store)


def test_terminal_append_cannot_change_prepared_payload(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="run-a")
    prepared = _intent(store.run_id)
    append_decision_entry(store, prepared, status="prepared")
    changed = prepared.model_copy(update={"note": "changed after preparation"})
    with pytest.raises(ValueError, match="terminal payload differs"):
        append_decision_entry(store, changed, status="applied")
    assert [entry.status for entry in read_decision_journal(store)] == ["prepared"]
