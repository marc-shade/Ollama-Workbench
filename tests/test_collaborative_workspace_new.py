"""
Tests for collaborative_workspace.py and its core dependency canvas.py.

Tests focus on DocumentState (pure Python logic) and avoid triggering the
Streamlit UI rendering functions (canvas_ui, collaborative_workspace_ui).
"""

import json
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixture: isolated canvas directory so tests don't pollute the repo
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_canvas_dir(tmp_path, monkeypatch):
    """
    Redirect the canvas storage directory to a temporary folder so tests
    don't write real files into the working directory.

    Strategy: import the module, then overwrite CANVAS_DIR on the module object.
    We must NOT reload after setting the attribute, or it resets to the default.
    """
    canvas_dir = tmp_path / "canvas"
    canvas_dir.mkdir()

    with patch.dict("sys.modules", {
        "streamlit": MagicMock(),
        "ollama_workbench.providers.ollama_utils": MagicMock(),
    }):
        import ollama_workbench.chat.canvas as canvas_mod
        # Override the module-level variable so DocumentState.save() writes here
        original_canvas_dir = canvas_mod.CANVAS_DIR
        canvas_mod.CANVAS_DIR = str(canvas_dir)
        try:
            yield canvas_mod, canvas_dir
        finally:
            canvas_mod.CANVAS_DIR = original_canvas_dir


# ---------------------------------------------------------------------------
# DocumentState — creation and initial state
# ---------------------------------------------------------------------------

class TestDocumentStateCreation:
    def test_new_document_has_empty_blocks(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState(name="My Doc")
        assert doc.blocks == []
        assert doc.version == 0
        assert doc.name == "My Doc"

    def test_doc_id_generated_when_not_provided(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        assert doc.doc_id is not None
        assert len(doc.doc_id) > 0

    def test_custom_doc_id_preserved(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState(doc_id="test-id-123")
        assert doc.doc_id == "test-id-123"

    def test_pending_ai_edits_initially_empty(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        assert doc.pending_ai_edits == []

    def test_edit_history_initially_empty(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        assert doc.edit_history == []


# ---------------------------------------------------------------------------
# DocumentState.add_block
# ---------------------------------------------------------------------------

class TestAddBlock:
    def test_add_text_block_increments_version(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        initial_version = doc.version
        doc.add_block("text", "Hello")
        assert doc.version == initial_version + 1

    def test_add_block_returns_id(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "content")
        assert isinstance(block_id, str)
        assert len(block_id) > 0

    def test_block_is_in_document_after_add(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "some content")
        assert len(doc.blocks) == 1
        assert doc.blocks[0]["id"] == block_id
        assert doc.blocks[0]["content"] == "some content"

    def test_add_multiple_blocks(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("text", "first")
        doc.add_block("code", "print('hello')")
        assert len(doc.blocks) == 2

    def test_block_type_stored_as_string_value(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("code", "x = 1")
        assert doc.blocks[0]["type"] == "code"

    def test_invalid_block_type_falls_back_to_text(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("NONEXISTENT_TYPE", "data")
        assert doc.blocks[0]["type"] == "text"

    def test_block_metadata_stored(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("code", "x = 1", metadata={"language": "python"})
        assert doc.blocks[0]["metadata"]["language"] == "python"

    def test_add_block_records_history(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("text", "hello")
        assert len(doc.edit_history) == 1
        assert doc.edit_history[0]["action"] == "add_block"


# ---------------------------------------------------------------------------
# DocumentState.update_block
# ---------------------------------------------------------------------------

class TestUpdateBlock:
    def test_update_existing_block(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "original")
        success = doc.update_block(block_id, "updated")
        assert success is True
        assert doc.get_block(block_id)["content"] == "updated"

    def test_update_nonexistent_block_returns_false(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        success = doc.update_block("does-not-exist", "new content")
        assert success is False

    def test_update_increments_version(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "v1")
        v_before = doc.version
        doc.update_block(block_id, "v2")
        assert doc.version == v_before + 1

    def test_update_stores_version_history(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "original")
        doc.update_block(block_id, "updated")
        block = doc.get_block(block_id)
        assert len(block["version_history"]) == 1
        assert block["version_history"][0]["content"] == "original"

    def test_version_history_capped_at_10(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "v0")
        for i in range(12):
            doc.update_block(block_id, f"v{i+1}")
        block = doc.get_block(block_id)
        assert len(block["version_history"]) <= 10


# ---------------------------------------------------------------------------
# DocumentState.delete_block
# ---------------------------------------------------------------------------

class TestDeleteBlock:
    def test_delete_existing_block(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "to delete")
        success = doc.delete_block(block_id)
        assert success is True
        assert len(doc.blocks) == 0

    def test_delete_nonexistent_block_returns_false(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        success = doc.delete_block("ghost-block")
        assert success is False

    def test_delete_records_history(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "bye")
        history_len_before = len(doc.edit_history)
        doc.delete_block(block_id)
        assert len(doc.edit_history) == history_len_before + 1
        assert doc.edit_history[-1]["action"] == "delete_block"


# ---------------------------------------------------------------------------
# DocumentState.move_block
# ---------------------------------------------------------------------------

class TestMoveBlock:
    def test_move_block_changes_position(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        id1 = doc.add_block("text", "first")
        id2 = doc.add_block("text", "second")
        id3 = doc.add_block("text", "third")

        # Move the first block to the end
        success = doc.move_block(id1, 2)
        assert success is True
        assert doc.blocks[2]["id"] == id1

    def test_move_nonexistent_block_returns_false(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("text", "a")
        success = doc.move_block("no-such-id", 0)
        assert success is False

    def test_move_to_same_position_returns_false(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "only")
        success = doc.move_block(block_id, 0)
        assert success is False

    def test_move_clamps_negative_position(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("text", "first")
        id2 = doc.add_block("text", "second")
        doc.move_block(id2, -99)
        assert doc.blocks[0]["id"] == id2


# ---------------------------------------------------------------------------
# DocumentState — AI suggestions
# ---------------------------------------------------------------------------

class TestAiSuggestions:
    def test_suggest_ai_edit_creates_pending_edit(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "original content")

        suggestion_id = doc.suggest_ai_edit(
            block_id,
            "improved content",
            canvas_mod.EditOperation.REPLACE,
            "Better phrasing"
        )

        assert len(doc.pending_ai_edits) == 1
        assert doc.pending_ai_edits[0]["id"] == suggestion_id
        assert doc.pending_ai_edits[0]["status"] == "pending"

    def test_apply_ai_suggestion_updates_block(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "old")

        suggestion_id = doc.suggest_ai_edit(
            block_id, "new", canvas_mod.EditOperation.REPLACE
        )
        success = doc.apply_ai_suggestion(suggestion_id)

        assert success is True
        assert doc.get_block(block_id)["content"] == "new"

    def test_apply_ai_suggestion_marks_status_accepted(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "before")
        suggestion_id = doc.suggest_ai_edit(
            block_id, "after", canvas_mod.EditOperation.REPLACE
        )
        doc.apply_ai_suggestion(suggestion_id)
        edit = next(e for e in doc.pending_ai_edits if e["id"] == suggestion_id)
        assert edit["status"] == "accepted"

    def test_reject_ai_suggestion_marks_status_rejected(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "original")
        suggestion_id = doc.suggest_ai_edit(
            block_id, "rejected version", canvas_mod.EditOperation.REPLACE
        )
        success = doc.reject_ai_suggestion(suggestion_id)

        assert success is True
        edit = next(e for e in doc.pending_ai_edits if e["id"] == suggestion_id)
        assert edit["status"] == "rejected"
        # Original content unchanged
        assert doc.get_block(block_id)["content"] == "original"

    def test_apply_nonexistent_suggestion_returns_false(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        result = doc.apply_ai_suggestion("made-up-id")
        assert result is False

    def test_reject_nonexistent_suggestion_returns_false(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        result = doc.reject_ai_suggestion("made-up-id")
        assert result is False

    def test_apply_already_accepted_suggestion_returns_false(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        block_id = doc.add_block("text", "text")
        suggestion_id = doc.suggest_ai_edit(
            block_id, "updated", canvas_mod.EditOperation.REPLACE
        )
        doc.apply_ai_suggestion(suggestion_id)
        # Try to apply again
        result = doc.apply_ai_suggestion(suggestion_id)
        assert result is False


# ---------------------------------------------------------------------------
# DocumentState.get_document_content
# ---------------------------------------------------------------------------

class TestGetDocumentContent:
    def test_empty_document_returns_empty_string(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        content = doc.get_document_content()
        assert content == ""

    def test_text_blocks_joined(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("text", "paragraph one")
        doc.add_block("text", "paragraph two")
        content = doc.get_document_content()
        assert "paragraph one" in content
        assert "paragraph two" in content

    def test_code_blocks_wrapped_in_fences(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("code", "x = 1", metadata={"language": "python"})
        content = doc.get_document_content()
        assert "```python" in content
        assert "x = 1" in content


# ---------------------------------------------------------------------------
# DocumentState.generate_diff
# ---------------------------------------------------------------------------

class TestGenerateDiff:
    def test_diff_of_identical_content_is_empty(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        diff = canvas_mod.DocumentState.generate_diff("same", "same")
        assert diff == ""

    def test_diff_shows_changed_line(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        diff = canvas_mod.DocumentState.generate_diff("old line", "new line")
        assert "-old line" in diff or "+new line" in diff


# ---------------------------------------------------------------------------
# DocumentState persistence (save / load)
# ---------------------------------------------------------------------------

class TestDocumentPersistence:
    def test_save_creates_json_file(self, tmp_canvas_dir):
        canvas_mod, canvas_dir = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("text", "persisted")
        # save() is called automatically; verify file exists
        expected_file = canvas_dir / f"{doc.doc_id}.json"
        assert expected_file.exists()

    def test_load_restores_blocks(self, tmp_canvas_dir):
        canvas_mod, canvas_dir = tmp_canvas_dir
        doc = canvas_mod.DocumentState()
        doc.add_block("text", "hello from disk")
        doc_id = doc.doc_id

        # Create a fresh DocumentState with the same id — it should load from disk
        doc2 = canvas_mod.DocumentState(doc_id=doc_id)
        assert len(doc2.blocks) == 1
        assert doc2.blocks[0]["content"] == "hello from disk"

    def test_load_nonexistent_doc_returns_empty(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        doc = canvas_mod.DocumentState(doc_id="nonexistent-id-xyz")
        assert doc.blocks == []


# ---------------------------------------------------------------------------
# EditOperation and EditStatus enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_edit_operation_values(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        assert canvas_mod.EditOperation.INSERT.value == "insert"
        assert canvas_mod.EditOperation.DELETE.value == "delete"
        assert canvas_mod.EditOperation.REPLACE.value == "replace"

    def test_edit_status_values(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        assert canvas_mod.EditStatus.PENDING.value == "pending"
        assert canvas_mod.EditStatus.ACCEPTED.value == "accepted"
        assert canvas_mod.EditStatus.REJECTED.value == "rejected"

    def test_block_type_values(self, tmp_canvas_dir):
        canvas_mod, _ = tmp_canvas_dir
        assert canvas_mod.BlockType.TEXT.value == "text"
        assert canvas_mod.BlockType.CODE.value == "code"
        assert canvas_mod.BlockType.MARKDOWN.value == "markdown"


# ---------------------------------------------------------------------------
# collaborative_workspace_ui import check
# ---------------------------------------------------------------------------

class TestCollaborativeWorkspaceImport:
    def test_module_imports_without_error(self):
        st_mock = MagicMock()
        st_mock.session_state = {}
        with patch.dict("sys.modules", {
            "streamlit": st_mock,
            "ollama_workbench.providers.ollama_utils": MagicMock(),
        }):
            import importlib
            import ollama_workbench.chat.collaborative_workspace as cw
            importlib.reload(cw)
            assert callable(cw.collaborative_workspace_ui)
