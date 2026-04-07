import json

from knowledge.store import KnowledgeStore


def test_load_nonexistent_creates_empty(tmp_path):
    store = KnowledgeStore(str(tmp_path / "knowledge.json"))
    data = store.load()
    assert data["version"] == "1.0"
    assert data["patterns"] == []


def test_load_existing_file(tmp_path):
    f = tmp_path / "knowledge.json"
    f.write_text(json.dumps({
        "version": "1.0",
        "lastUpdated": "",
        "patterns": [{"id": "p1", "symptom": "test"}],
    }))
    store = KnowledgeStore(str(f))
    data = store.load()
    assert len(data["patterns"]) == 1
    assert data["patterns"][0]["id"] == "p1"


def test_load_caches_by_mtime(tmp_path):
    f = tmp_path / "knowledge.json"
    f.write_text(json.dumps({"version": "1.0", "lastUpdated": "", "patterns": []}))
    store = KnowledgeStore(str(f))
    d1 = store.load()
    d2 = store.load()
    assert d1 is d2  # same object = cache hit


def _make_pattern(**overrides):
    base = {
        "id": "pattern_test1",
        "symptom": "字段不显示",
        "symptom_keywords": ["字段", "不显示", "看不到"],
        "misdiagnosis_trap": "容易误判为 schema 配置",
        "actual_root_cause": "runtime 逻辑",
        "root_cause_type": "runtime_business_logic",
        "diagnostic_shortcut": "先查 use-model.ts",
        "key_files": ["use-model.ts"],
        "resolution": "检查条件渲染",
        "confidence": 0.9,
        "source_cases": ["case1"],
        "times_matched": 0,
        "createdAt": "2026-04-08T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_add_pattern_new(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern())
    data = store.load()
    assert len(data["patterns"]) == 1
    assert data["patterns"][0]["id"] == "pattern_test1"
    assert data["lastUpdated"] != ""


def test_add_pattern_dedup_merges(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern(confidence=0.8, source_cases=["case1"]))
    store.add_pattern(_make_pattern(
        id="pattern_test2",
        confidence=0.95,
        source_cases=["case2"],
        symptom_keywords=["字段", "不显示", "隐藏"],
    ))
    data = store.load()
    assert len(data["patterns"]) == 1  # merged, not two
    assert data["patterns"][0]["confidence"] == 0.95
    assert "case2" in data["patterns"][0]["source_cases"]
    assert "隐藏" in data["patterns"][0]["symptom_keywords"]


def test_add_pattern_different_type_no_merge(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern(root_cause_type="runtime_business_logic"))
    store.add_pattern(_make_pattern(
        id="pattern_test2",
        root_cause_type="schema_config",
    ))
    data = store.load()
    assert len(data["patterns"]) == 2


def test_match_keyword_overlap(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern())
    results = store.match("商家反馈字段不显示了")
    assert len(results) == 1
    assert results[0]["id"] == "pattern_test1"
    assert results[0]["times_matched"] == 1


def test_match_no_overlap_returns_empty(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    store.add_pattern(_make_pattern())
    results = store.match("价格计算错误")
    assert results == []


def test_match_respects_top_k(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.json"))
    for i in range(5):
        store.add_pattern(_make_pattern(
            id=f"p{i}",
            root_cause_type=f"type_{i}",  # distinct types so no merge
            confidence=0.5 + i * 0.1,
        ))
    results = store.match("字段不显示", top_k=2)
    assert len(results) == 2
