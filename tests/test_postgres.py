import os
import sys
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from db.postgres import PostgresDB


@pytest.mark.skipif(
    os.getenv("POSTGRES_DSN") is None,
    reason="POSTGRES_DSN not set",
)
def test_postgres_connection():
    db = PostgresDB(os.getenv("POSTGRES_DSN"))
    assert db.conn is not None
