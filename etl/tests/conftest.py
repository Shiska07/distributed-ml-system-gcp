import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def spark():
    """One local Spark session for all tests (starting Spark is slow)."""
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    # Spark's Python workers are separate processes: give them the same Python and import path
    etl_dir = str(Path(__file__).resolve().parents[1])
    os.environ["PYTHONPATH"] = etl_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYSPARK_PYTHON"] = sys.executable

    session = (SparkSession.builder.master("local[1]").appName("tests")
               .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    yield session
    session.stop()
    