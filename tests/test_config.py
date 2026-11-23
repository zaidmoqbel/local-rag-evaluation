from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_config_paths_follow_yaml(tmp_path: Path) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_bytes(CONFIG.read_bytes())
    config = load_config(settings)
    assert config.paths.raw_data_dir == tmp_path / "data" / "raw"
    assert config.paths.vector_store_dir == tmp_path / "vector_store"


def test_config_rejects_nonadvancing_windows(tmp_path: Path) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "chunk_overlap: 150", "chunk_overlap: 1000"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        load_config(settings)
