import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from middlewares.knowledge import KnowledgeMiddleware


def _msg(type: str, content: str, tool_calls=None):
    m = MagicMock()
    m.type = type
    m.content = content
    m.tool_calls = tool_calls
    return m


class TestRatingDetection:
    def _mw(self, tmp_path):
        return KnowledgeMiddleware(knowledge_path=str(tmp_path / "k.json"))

    def test_detect_score_after_rating_prompt(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("human", "字段不显示"),
            _msg("ai", "根因是 runtime 逻辑。请对本次诊断评分（1-10 分）"),
            _msg("human", "8"),
        ]
        score, comment = mw._detect_rating(messages)
        assert score == 8
        assert comment == ""

    def test_detect_score_with_comment(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("ai", "请评分"),
            _msg("human", "9 这次定位很快"),
        ]
        score, comment = mw._detect_rating(messages)
        assert score == 9
        assert comment == "这次定位很快"

    def test_no_rating_prompt_returns_none(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("ai", "根因是 runtime 逻辑"),
            _msg("human", "8"),  # number but no rating prompt
        ]
        score, comment = mw._detect_rating(messages)
        assert score is None

    def test_score_out_of_range_returns_none(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("ai", "请评分"),
            _msg("human", "15"),
        ]
        score, comment = mw._detect_rating(messages)
        assert score is None

    def test_chinese_fen_suffix(self, tmp_path):
        mw = self._mw(tmp_path)
        messages = [
            _msg("ai", "请评分"),
            _msg("human", "8分"),
        ]
        score, comment = mw._detect_rating(messages)
        assert score == 8


class TestAfterAgent:
    def test_extracts_on_high_score(self, tmp_path):
        kpath = str(tmp_path / "k.json")
        mw = KnowledgeMiddleware(knowledge_path=kpath, score_threshold=7)

        state = {
            "messages": [
                _msg("human", "字段不显示怎么办"),
                _msg("ai", "根因是 runtime 逻辑。请评分"),
                _msg("human", "8 很准确"),
            ]
        }

        mock_pattern = {
            "symptom": "字段不显示",
            "symptom_keywords": ["字段", "不显示"],
            "root_cause_type": "runtime_business_logic",
            "confidence": 0.9,
        }

        with patch("knowledge.extractor.extract_pattern", return_value=mock_pattern) as mock_extract:
            mw.after_agent(state, runtime=MagicMock())
            mock_extract.assert_called_once()

        data = json.loads(Path(kpath).read_text())
        assert len(data["patterns"]) == 1

    def test_skips_low_score(self, tmp_path):
        mw = KnowledgeMiddleware(
            knowledge_path=str(tmp_path / "k.json"), score_threshold=7
        )
        state = {
            "messages": [
                _msg("ai", "请评分"),
                _msg("human", "3"),
            ]
        }
        with patch("knowledge.extractor.extract_pattern") as mock_extract:
            mw.after_agent(state, runtime=MagicMock())
            mock_extract.assert_not_called()


class TestBeforeModel:
    def test_injects_on_first_human_message(self, tmp_path):
        kpath = tmp_path / "k.json"
        kpath.write_text(json.dumps({
            "version": "1.0",
            "lastUpdated": "",
            "patterns": [{
                "id": "p1",
                "symptom": "字段不显示",
                "symptom_keywords": ["字段", "不显示"],
                "misdiagnosis_trap": "误判为 schema",
                "actual_root_cause": "runtime",
                "root_cause_type": "runtime_business_logic",
                "diagnostic_shortcut": "先查 use-model.ts",
                "key_files": ["use-model.ts"],
                "resolution": "检查条件渲染",
                "confidence": 0.95,
                "source_cases": ["case1"],
                "times_matched": 0,
                "createdAt": "2026-04-08T00:00:00Z",
            }],
        }))

        mw = KnowledgeMiddleware(knowledge_path=str(kpath))
        sys_msg = _msg("system", "你是 oncall 助手")
        state = {
            "messages": [
                sys_msg,
                _msg("human", "商家反馈字段不显示"),
            ]
        }
        result = mw.before_model(state, runtime=MagicMock())
        assert result is not None
        sys_content = result["messages"][0].content
        assert "<diagnostic_knowledge>" in sys_content
        assert "use-model.ts" in sys_content

    def test_skips_after_first_turn(self, tmp_path):
        kpath = tmp_path / "k.json"
        kpath.write_text(json.dumps({
            "version": "1.0", "lastUpdated": "",
            "patterns": [{"id": "p1", "symptom_keywords": ["字段"]}],
        }))
        mw = KnowledgeMiddleware(knowledge_path=str(kpath))
        state = {
            "messages": [
                _msg("system", "你是助手"),
                _msg("human", "字段不显示"),
                _msg("ai", "我来查"),
                _msg("human", "是水果类目的"),  # second human msg
            ]
        }
        result = mw.before_model(state, runtime=MagicMock())
        assert result is None
