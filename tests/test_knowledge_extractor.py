import json
from unittest.mock import MagicMock, patch

from knowledge.extractor import extract_pattern, _format_messages, _parse_llm_json


def test_format_messages():
    """Messages are formatted as 'role: content' lines."""
    msgs = [
        MagicMock(type="human", content="字段不显示"),
        MagicMock(type="ai", content="我来帮你看看", tool_calls=None),
    ]
    text = _format_messages(msgs)
    assert "用户: 字段不显示" in text
    assert "助手: 我来帮你看看" in text


def test_parse_llm_json_valid():
    raw = '```json\n{"symptom": "test", "confidence": 0.9}\n```'
    result = _parse_llm_json(raw)
    assert result["symptom"] == "test"


def test_parse_llm_json_null():
    assert _parse_llm_json("null") is None


def test_parse_llm_json_invalid():
    assert _parse_llm_json("not json at all") is None


def test_extract_pattern_happy_path():
    llm_response = json.dumps({
        "symptom": "字段不显示",
        "symptom_keywords": ["字段", "不显示"],
        "misdiagnosis_trap": "误判为 schema",
        "actual_root_cause": "runtime 逻辑",
        "root_cause_type": "runtime_business_logic",
        "diagnostic_shortcut": "先查 use-model.ts",
        "key_files": ["use-model.ts"],
        "resolution": "检查条件渲染",
        "confidence": 0.9,
    })
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content=llm_response)

    with patch("knowledge.extractor.create_chat_model", return_value=mock_model):
        msgs = [MagicMock(type="human", content="字段不显示")]
        result = extract_pattern(msgs, score=8, comment="不错")

    assert result is not None
    assert result["symptom"] == "字段不显示"
    assert result["id"].startswith("pattern_")
    assert result["times_matched"] == 0
    assert result["source_cases"] == ["字段不显示"]


def test_extract_pattern_null_response():
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="null")

    with patch("knowledge.extractor.create_chat_model", return_value=mock_model):
        msgs = [MagicMock(type="human", content="hello")]
        result = extract_pattern(msgs, score=8)

    assert result is None
