from backend.chatbot import context


def test_read_template_source_includes_real_transformer_code():
    source = context._read_template_source("transformer")
    assert "class RotaryPositionalEncoding" in source
    assert "model.py" in source
    assert "data.py" in source


def test_read_template_source_unknown_template_does_not_crash():
    source = context._read_template_source("nonexistent_template")
    assert "No source found" in source
