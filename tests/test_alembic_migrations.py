"""Alembic migration tests (PLAN.md PR-2a).

Upgrades a throwaway SQLite DB to head and checks every table exists.
"""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect

from src.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = {
    "users",
    "workspaces",
    "documents",
    "answers",
    "answer_traces",
    "usage_events",
    "subscriptions",
    "alembic_version",
}


@pytest.fixture()
def alembic_cfg(tmp_path, monkeypatch):
    """Alembic config pointed at a fresh SQLite file (Config.DATABASE_URL overridden)."""
    url = f"sqlite:///{tmp_path / 'migration.db'}"
    monkeypatch.setattr(Config, "DATABASE_URL", url)
    cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg, url


def test_upgrade_head_creates_all_tables(alembic_cfg):
    cfg, url = alembic_cfg

    command.upgrade(cfg, "head")

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert tables == EXPECTED_TABLES


def test_downgrade_base_drops_all_tables(alembic_cfg):
    cfg, url = alembic_cfg
    command.upgrade(cfg, "head")

    command.downgrade(cfg, "base")

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    # Alembic keeps its own bookkeeping table; all app tables must be gone.
    assert tables == {"alembic_version"}
