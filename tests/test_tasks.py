from e2m.harness.starter import STARTER_TASKS, write_starter_jsonl
from e2m.harness.tasks import load_jsonl, starter_jsonl_path


def test_roundtrip_jsonl(tmp_path):
    path = tmp_path / "t.jsonl"
    write_starter_jsonl(path)
    loaded = load_jsonl(path)
    assert len(loaded) == len(STARTER_TASKS)
    assert [t.id for t in loaded] == [t.id for t in STARTER_TASKS]
    for t in loaded:
        assert t.spec
        assert t.signature
        assert t.reference_c
        assert t.tests
        t.sig()  # parses


def test_repo_jsonl_written():
    write_starter_jsonl(starter_jsonl_path())
    assert starter_jsonl_path().is_file()
    assert len(load_jsonl(starter_jsonl_path())) >= 20
