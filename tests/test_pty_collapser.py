"""Tests for the refactored PTYCollapser (raw-message-based)."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cta.pty_collapser import collapse_pty_sessions, h3_verdict, PTYSession
from cta.hermes_adapter import hermes_session_to_trace
from cta.data_models import EventType, EventOutcome


# --------------------------------------------------------------------------- #
# Synthetic M3 fixture: mirrors the documented interactive timeline
# --------------------------------------------------------------------------- #

def _tc(call_id, name, arguments):
    """Build a tool_call block."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _assistant(tool_calls, reasoning="", content=""):
    return {
        "role": "assistant",
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
    }


def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def make_interactive_session():
    """Build a synthetic Hermes session mirroring the M3 interactive timeline."""
    messages = [
        # msg 4: binary resolution
        _assistant([_tc("tc1", "terminal", {"command": "which -a qodercli && qodercli --version"})]),
        _tool_result("tc1", "/usr/local/bin/qodercli\nqodercli v1.1.1"),

        # msg 10: PTY parent launch
        _assistant([_tc("tc2", "terminal", {
            "command": "qodercli -i 'Implement auth across src/routes, src/models, tests'",
            "pty": True,
            "background": True,
            "timeout": 300,
        })], reasoning="Delegating to qodercli interactive mode"),
        _tool_result("tc2", json.dumps({"output": "Background process started", "session_id": "proc_abc123"})),

        # msg 12: first poll
        _assistant([_tc("tc3", "process", {"action": "poll", "session_id": "proc_abc123"})]),
        _tool_result("tc3", "Checking folder trust..."),

        # msg 14: log
        _assistant([_tc("tc4", "process", {"action": "log", "session_id": "proc_abc123"})]),
        _tool_result("tc4", "Do you trust this folder? [1] Yes [2] No"),

        # msg 16: TRUST DIALOG RESOLVED
        _assistant([_tc("tc5", "process", {"action": "submit", "data": "1", "session_id": "proc_abc123"})]),
        _tool_result("tc5", "Input sent."),

        # msg 18: interleaved read_file (model checks output between process calls)
        _assistant([_tc("tc6", "read_file", {"path": "src/routes/auth.py"})]),
        _tool_result("tc6", "# auth route placeholder"),

        # msg 20: poll
        _assistant([_tc("tc7", "process", {"action": "poll", "session_id": "proc_abc123"})]),
        _tool_result("tc7", "Working on src/models/user.py..."),

        # msg 22: wait
        _assistant([_tc("tc8", "process", {"action": "wait", "session_id": "proc_abc123"})]),
        _tool_result("tc8", "Process still running."),

        # msg 39: permission prompt #1
        _assistant([_tc("tc9", "process", {"action": "submit", "data": "1", "session_id": "proc_abc123"})]),
        _tool_result("tc9", "Input sent."),

        # msg 48: permission prompt #2
        _assistant([_tc("tc10", "process", {"action": "submit", "data": "1", "session_id": "proc_abc123"})]),
        _tool_result("tc10", "Input sent."),

        # msg 55: final poll
        _assistant([_tc("tc11", "process", {"action": "poll", "session_id": "proc_abc123"})]),
        _tool_result("tc11", "Task complete. 4 files modified."),

        # msg 58: kill
        _assistant([_tc("tc12", "process", {"action": "kill", "session_id": "proc_abc123"})]),
        _tool_result("tc12", "Process terminated."),
    ]
    return {"session_id": "test_interactive", "model": "test", "messages": messages}


def make_non_pty_session():
    """Build a session with no PTY usage (print-mode style)."""
    messages = [
        _assistant([_tc("tc1", "terminal", {"command": "which -a qodercli && qodercli --version"})]),
        _tool_result("tc1", "/usr/local/bin/qodercli"),

        _assistant([_tc("tc2", "terminal", {
            "command": "qodercli -p 'Implement tax helper' --print",
            "timeout": 180,
        })], reasoning="Using print mode for bounded task"),
        _tool_result("tc2", "Created src/utils/tax.py"),

        _assistant([_tc("tc3", "read_file", {"path": "src/utils/tax.py"})]),
        _tool_result("tc3", "def calculate_tax(...): ..."),

        # Pure reasoning turn
        {"role": "assistant", "content": "Task looks complete.", "reasoning": "Verifying output", "tool_calls": []},
    ]
    return {"session_id": "test_print_mode", "model": "test", "messages": messages}


# --------------------------------------------------------------------------- #
# Tests: Interactive PTY session
# --------------------------------------------------------------------------- #

class TestInteractivePTY:
    def setup_method(self):
        self.session = make_interactive_session()
        self.trace, self.pty_sessions, self.report = collapse_pty_sessions(
            self.session["messages"], with_skill=True, trace_id="test_interactive"
        )

    def test_one_pty_session_detected(self):
        assert len(self.pty_sessions) == 1

    def test_session_id_extracted(self):
        assert self.pty_sessions[0].session_id == "proc_abc123"

    def test_command_preserved(self):
        assert "qodercli -i" in self.pty_sessions[0].command

    def test_poll_count(self):
        # poll×3 + wait×1 = 4 (wait counts as poll)
        assert self.pty_sessions[0].total_polls == 4

    def test_write_count(self):
        # submit×3 (trust + 2 permissions)
        assert self.pty_sessions[0].total_writes == 3

    def test_log_count(self):
        assert self.pty_sessions[0].total_logs == 1

    def test_trust_dialog_handled(self):
        assert self.pty_sessions[0].trust_dialog_handled is True

    def test_killed(self):
        assert self.pty_sessions[0].killed is True

    def test_composite_event_is_single_execute(self):
        execute_events = [e for e in self.trace.events if e.type == EventType.EXECUTE]
        # Original: terminal(which) + terminal(qodercli -i) = 2 EXECUTE
        # After collapse: terminal(which) + composite = 2 EXECUTE
        # The 10 process() calls are consumed into the composite
        assert len(execute_events) == 2

    def test_composite_content_is_json(self):
        execute_events = [e for e in self.trace.events if e.type == EventType.EXECUTE]
        composite = execute_events[1]  # second EXECUTE is the composite
        parsed = json.loads(composite.content)
        assert parsed["session_id"] == "proc_abc123"
        assert parsed["trust_dialog_handled"] is True
        assert parsed["terminated"] == "kill"
        assert len(parsed["actions"]) == 9  # 9 process calls (tc6 is read_file)

    def test_interleaved_read_file_survives(self):
        read_events = [e for e in self.trace.events if e.type == EventType.READ]
        assert len(read_events) == 1
        assert "auth.py" in read_events[0].target

    def test_process_calls_consumed(self):
        tool_call_events = [e for e in self.trace.events if e.type == EventType.TOOL_CALL]
        # All process() calls should be consumed (none remain as TOOL_CALL)
        assert len(tool_call_events) == 0

    def test_no_none_events(self):
        assert self.report["none_events"] == 0

    def test_pty_collapsed_calls_count(self):
        assert self.report["pty_collapsed_calls"] == 9

    def test_h3_confirmed(self):
        verdict = h3_verdict(self.pty_sessions)
        assert verdict["status"] == "CONFIRMED"

    def test_child_call_ids_recorded(self):
        session = self.pty_sessions[0]
        assert len(session.child_call_ids) == 9
        assert "tc3" in session.child_call_ids
        assert "tc12" in session.child_call_ids


# --------------------------------------------------------------------------- #
# Tests: Non-PTY parity (output matches adapter)
# --------------------------------------------------------------------------- #

class TestNonPTYPassthrough:
    def setup_method(self):
        self.session = make_non_pty_session()
        self.trace, self.pty_sessions, self.report = collapse_pty_sessions(
            self.session["messages"], with_skill=True, trace_id="test_print_mode"
        )
        self.adapter_trace, _ = hermes_session_to_trace(self.session, with_skill=True)

    def test_no_pty_sessions(self):
        assert len(self.pty_sessions) == 0

    def test_event_count_matches_adapter(self):
        # Both should produce the same number of events
        # (3 tool calls + 1 reason turn = 4 events)
        assert len(self.trace.events) == len(self.adapter_trace.events)

    def test_event_types_match_adapter(self):
        collapser_types = [e.type for e in self.trace.events]
        adapter_types = [e.type for e in self.adapter_trace.events]
        assert collapser_types == adapter_types

    def test_event_targets_match_adapter(self):
        collapser_targets = [e.target for e in self.trace.events]
        adapter_targets = [e.target for e in self.adapter_trace.events]
        assert collapser_targets == adapter_targets

    def test_event_content_match_adapter(self):
        collapser_content = [e.content for e in self.trace.events]
        adapter_content = [e.content for e in self.adapter_trace.events]
        assert collapser_content == adapter_content

    def test_h3_untestable(self):
        verdict = h3_verdict(self.pty_sessions)
        assert verdict["status"] == "UNTESTABLE"


# --------------------------------------------------------------------------- #
# Tests: Edge cases
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_empty_messages(self):
        trace, sessions, report = collapse_pty_sessions([])
        assert len(trace.events) == 0
        assert len(sessions) == 0
        assert report["event_count"] == 0

    def test_pty_parent_without_children(self):
        """PTY launch with no subsequent process() calls."""
        messages = [
            _assistant([_tc("tc1", "terminal", {
                "command": "qodercli -i 'do stuff'",
                "pty": True,
                "background": True,
            })]),
            _tool_result("tc1", json.dumps({"output": "Background process started", "session_id": "proc_x"})),
            _assistant([_tc("tc2", "read_file", {"path": "foo.py"})]),
            _tool_result("tc2", "content"),
        ]
        trace, sessions, report = collapse_pty_sessions(messages)
        assert len(sessions) == 1
        assert sessions[0].total_polls == 0
        assert sessions[0].killed is False
        # Composite still emitted (with PARTIAL outcome)
        execute_events = [e for e in trace.events if e.type == EventType.EXECUTE]
        assert len(execute_events) == 1
        assert execute_events[0].outcome == EventOutcome.PARTIAL

    def test_terminal_without_pty_not_collapsed(self):
        """terminal with qodercli but pty=False should NOT be treated as PTY parent."""
        messages = [
            _assistant([_tc("tc1", "terminal", {
                "command": "qodercli -p 'quick task' --print",
                "pty": False,
            })]),
            _tool_result("tc1", "Done."),
        ]
        trace, sessions, _ = collapse_pty_sessions(messages)
        assert len(sessions) == 0
        assert len(trace.events) == 1
        assert trace.events[0].type == EventType.EXECUTE

    def test_terminal_without_background_not_collapsed(self):
        """terminal with pty=True but background=False is foreground — not a PTY parent."""
        messages = [
            _assistant([_tc("tc1", "terminal", {
                "command": "qodercli -i 'task'",
                "pty": True,
                "background": False,
            })]),
            _tool_result("tc1", "output"),
        ]
        trace, sessions, _ = collapse_pty_sessions(messages)
        assert len(sessions) == 0

    def test_blockade_detection(self):
        """5+ polls without any writes triggers blockade in h3_verdict."""
        messages = [
            _assistant([_tc("tc1", "terminal", {
                "command": "qodercli -i 'task'",
                "pty": True,
                "background": True,
            })]),
            _tool_result("tc1", json.dumps({"output": "Background process started", "session_id": "proc_b"})),
        ]
        # 5 polls, no writes
        for i in range(5):
            messages.append(_assistant([_tc(f"p{i}", "process", {"action": "poll", "session_id": "proc_b"})]))
            messages.append(_tool_result(f"p{i}", "still waiting..."))

        _, sessions, _ = collapse_pty_sessions(messages)
        verdict = h3_verdict(sessions)
        assert verdict["status"] == "DISCONFIRMED"
        assert verdict["sessions"][0]["blockade_detected"] is True

    def test_error_in_child_sets_has_error(self):
        """If a process() observation contains error markers, session.has_error is set."""
        messages = [
            _assistant([_tc("tc1", "terminal", {
                "command": "qodercli -i 'task'",
                "pty": True,
                "background": True,
            })]),
            _tool_result("tc1", json.dumps({"output": "Background process started", "session_id": "proc_e"})),
            _assistant([_tc("tc2", "process", {"action": "poll", "session_id": "proc_e"})]),
            _tool_result("tc2", "Error: HTTP 402 credit limit exceeded"),
            _assistant([_tc("tc3", "process", {"action": "kill", "session_id": "proc_e"})]),
            _tool_result("tc3", "Process terminated."),
        ]
        _, sessions, _ = collapse_pty_sessions(messages)
        assert sessions[0].has_error is True


# --------------------------------------------------------------------------- #
# Integration test: real M3 capture (skipif db not present)
# --------------------------------------------------------------------------- #

M3_DB = Path(__file__).resolve().parent.parent / "data" / "m3_captures" / "P1-interactive-treatment-1" / "state.db"


def _load_session_from_db(db_path: Path) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    session_row = cur.execute("SELECT id, model FROM sessions LIMIT 1").fetchone()
    if not session_row:
        conn.close()
        return {}

    session_id = session_row["id"]
    rows = cur.execute(
        """SELECT id, role, content, tool_call_id, tool_calls, tool_name,
                  reasoning, reasoning_content, finish_reason
           FROM messages
           WHERE session_id = ? AND active = 1 AND compacted = 0
           ORDER BY id""",
        (session_id,),
    ).fetchall()
    conn.close()

    messages = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"] or "", "tool_call_id": r["tool_call_id"]}
        if r["role"] == "assistant":
            msg["reasoning"] = r["reasoning"] or r["reasoning_content"] or ""
            if r["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(r["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    msg["tool_calls"] = []
            else:
                msg["tool_calls"] = []
        messages.append(msg)

    return {"session_id": session_id, "model": session_row["model"], "messages": messages}


@pytest.mark.skipif(not M3_DB.exists(), reason="M3 capture state.db not available")
class TestM3Integration:
    def setup_method(self):
        self.session = _load_session_from_db(M3_DB)
        assert self.session, "Failed to load M3 session"
        self.trace, self.pty_sessions, self.report = collapse_pty_sessions(
            self.session["messages"], with_skill=True, trace_id="m3_integration"
        )

    def test_at_least_one_pty_session(self):
        assert len(self.pty_sessions) >= 1

    def test_trust_dialog_handled(self):
        assert any(s.trust_dialog_handled for s in self.pty_sessions)

    def test_multiple_writes(self):
        total_writes = sum(s.total_writes for s in self.pty_sessions)
        assert total_writes >= 4  # trust + 3 permission prompts

    def test_h3_confirmed(self):
        verdict = h3_verdict(self.pty_sessions)
        assert verdict["status"] == "CONFIRMED"

    def test_no_none_events(self):
        assert self.report["none_events"] == 0

    def test_process_calls_consumed(self):
        tool_call_events = [e for e in self.trace.events if e.type == EventType.TOOL_CALL]
        # process calls should be consumed into composites
        # (other TOOL_CALL types like delegate_task may remain)
        for e in tool_call_events:
            assert "process" not in e.target
