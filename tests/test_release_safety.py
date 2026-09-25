from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_HAS_COMPETITION_DATA = (_ROOT / "case-set.json").exists()


@pytest.mark.skipif(
    _HAS_COMPETITION_DATA,
    reason="competition payload present — this guard is for pre-release / CI only",
)
def test_repository_contains_no_competition_payload() -> None:
    root = _ROOT
    assert not (root / "case-set.json").exists()
    assert list((root / "inputs").glob("*.json")) == []
    assert list((root / "outputs").glob("*.json")) == []
    forbidden = {"oracles", "reference-outputs", "private-partitions.json", "mcp-access.json"}
    assert not any(path.name in forbidden for path in root.rglob("*"))


def test_example_environment_has_no_real_key() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / ".env.example").read_text(encoding="utf-8")
    assert "sk-team-replace_me" in content
    assert content.count("sk-team-") == 1
