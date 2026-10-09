-- 03_validate_silver.sql
-- Checks one Beam run in silver before Spark uses it: no duplicates, no nulls in key columns,
-- values in range, dates inside the run's window.
-- Each check is an ASSERT, so the script stops with an error at the first failed check
-- (a scheduled workflow can then stop the pipeline).
--
-- ONE-TIME SETUP: create one external table per dataset (rerun if the folder layout changes).
-- "WITH PARTITION COLUMNS" turns the folder names source=... and run_id=... into columns.
--
-- CREATE OR REPLACE EXTERNAL TABLE `chicago-taxi-tips-project.taxi_lake.silver_trips`
-- WITH PARTITION COLUMNS
-- OPTIONS (
--   format = 'PARQUET',
--   uris = ['gs://chicago-taxi-tips-project-lake/silver/trips/*'],
--   hive_partition_uri_prefix = 'gs://chicago-taxi-tips-project-lake/silver/trips'
-- );
--
-- CREATE OR REPLACE EXTERNAL TABLE `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
-- WITH PARTITION COLUMNS
-- OPTIONS (
--   format = 'PARQUET',
--   uris = ['gs://chicago-taxi-tips-project-lake/silver/outcomes/*'],
--   hive_partition_uri_prefix = 'gs://chicago-taxi-tips-project-lake/silver/outcomes'
-- );
--
-- List the available runs:
--   SELECT DISTINCT source, run_id FROM `chicago-taxi-tips-project.taxi_lake.silver_trips` ORDER BY run_id;
--   SELECT DISTINCT source, run_id FROM `chicago-taxi-tips-project.taxi_lake.silver_outcomes` ORDER BY run_id;


-- ---------------- RUN TO CHECK (printed by the pipeline as ">> Run ID: ...") ----------------
DECLARE target_source STRING DEFAULT 'historical';          -- historical | live
DECLARE target_run    STRING DEFAULT '2022-01-01_2022-01-02_20261010-120000';
DECLARE check_trips    BOOL DEFAULT TRUE;                  -- set FALSE if this run had no trips
DECLARE check_outcomes BOOL DEFAULT TRUE;                  -- set FALSE if this run had no outcomes

-- the run's date window comes from the run_id itself: <start>_<end>_<timestamp>
DECLARE run_start DATE DEFAULT DATE(SPLIT(target_run, '_')[OFFSET(0)]);
DECLARE run_end   DATE DEFAULT DATE(SPLIT(target_run, '_')[OFFSET(1)]);


-- ---------------- TRIPS ----------------
IF check_trips THEN

  -- summary (shown in the results)
  SELECT
    COUNT(*)                         AS n_rows,
    COUNT(DISTINCT unique_key)       AS n_keys,
    COUNTIF(trip_start_ts IS NULL)   AS null_start,
    COUNTIF(fare IS NULL)            AS null_fare,
    MIN(DATE(trip_start_ts))         AS first_day,
    MAX(DATE(trip_start_ts))         AS last_day
  FROM `chicago-taxi-tips-project.taxi_lake.silver_trips`
  WHERE source = target_source AND run_id = target_run;

  ASSERT (SELECT COUNT(*) FROM `chicago-taxi-tips-project.taxi_lake.silver_trips`
          WHERE source = target_source AND run_id = target_run) > 0
    AS 'trips: run is empty or the run_id is wrong';

  ASSERT (SELECT COUNT(*) = COUNT(DISTINCT unique_key)
          FROM `chicago-taxi-tips-project.taxi_lake.silver_trips`
          WHERE source = target_source AND run_id = target_run)
    AS 'trips: duplicate unique_key';

  ASSERT (SELECT COUNTIF(unique_key IS NULL OR trip_start_ts IS NULL OR fare IS NULL
                         OR trip_seconds IS NULL OR trip_miles IS NULL) = 0
          FROM `chicago-taxi-tips-project.taxi_lake.silver_trips`
          WHERE source = target_source AND run_id = target_run)
    AS 'trips: nulls in required columns';

  ASSERT (SELECT COUNTIF(DATE(trip_start_ts) < run_start OR DATE(trip_start_ts) >= run_end) = 0
          FROM `chicago-taxi-tips-project.taxi_lake.silver_trips`
          WHERE source = target_source AND run_id = target_run)
    AS 'trips: trip dates outside the run window';

END IF;


-- ---------------- OUTCOMES ----------------
IF check_outcomes THEN

  SELECT
    COUNT(*)                     AS n_rows,
    COUNT(DISTINCT unique_key)   AS n_keys,
    COUNTIF(tips IS NULL)        AS null_tips,
    COUNTIF(tips < 0)            AS negative_tips
  FROM `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
  WHERE source = target_source AND run_id = target_run;

  ASSERT (SELECT COUNT(*) FROM `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
          WHERE source = target_source AND run_id = target_run) > 0
    AS 'outcomes: run is empty or the run_id is wrong';

  ASSERT (SELECT COUNT(*) = COUNT(DISTINCT unique_key)
          FROM `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
          WHERE source = target_source AND run_id = target_run)
    AS 'outcomes: duplicate unique_key';

  ASSERT (SELECT COUNTIF(unique_key IS NULL OR tips IS NULL OR tips < 0) = 0
          FROM `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
          WHERE source = target_source AND run_id = target_run)
    AS 'outcomes: null or negative tips';

END IF;


-- ---------------- MATCH RATE (information only) ----------------
-- Share of trips in this run that have an outcome in ANY run of the same source.
-- Historical: should be ~100%. Live: lower if outcomes haven't arrived yet.
IF check_trips THEN
  SELECT
    COUNT(*)                                        AS trips,
    COUNTIF(o.unique_key IS NOT NULL)               AS with_outcome,
    ROUND(COUNTIF(o.unique_key IS NOT NULL) / COUNT(*), 4) AS match_rate
  FROM `chicago-taxi-tips-project.taxi_lake.silver_trips` t
  LEFT JOIN (SELECT DISTINCT unique_key
             FROM `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
             WHERE source = target_source) o
  USING (unique_key)
  WHERE t.source = target_source AND t.run_id = target_run;
END IF;
