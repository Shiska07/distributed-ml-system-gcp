-- 03_validate_silver.sql
-- Checks ONE Beam run in silver before it is used downstream.
--   trips:    not empty, no duplicate keys, no missing required values, sensible values,
--             trip dates inside the run's date window
--   outcomes: not empty, no duplicate keys, tips present and not negative
-- Every check is an ASSERT: the script stops with an error at the first failure.
--
-- HOW TO USE: set the 4 values in "RUN TO CHECK" and run the whole script.
--
-- ONE-TIME SETUP (rerun only if the folder layout changes):
-- External tables over silver. Folder names source=… and run_id=… become columns.
--
-- CREATE OR REPLACE EXTERNAL TABLE `chicago-taxi-tips-project.taxi_lake.silver_trips`
-- WITH PARTITION COLUMNS
-- OPTIONS (format = 'PARQUET',
--          uris = ['gs://chicago-taxi-tips-project-lake/silver/trips/*.parquet'],
--          hive_partition_uri_prefix = 'gs://chicago-taxi-tips-project-lake/silver/trips');
--
-- CREATE OR REPLACE EXTERNAL TABLE `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
-- WITH PARTITION COLUMNS
-- OPTIONS (format = 'PARQUET',
--          uris = ['gs://chicago-taxi-tips-project-lake/silver/outcomes/*.parquet'],
--          hive_partition_uri_prefix = 'gs://chicago-taxi-tips-project-lake/silver/outcomes');
--
-- Find a run_id:
--   SELECT DISTINCT source, run_id FROM `chicago-taxi-tips-project.taxi_lake.silver_trips` ORDER BY run_id;


-- ======================= RUN TO CHECK =======================
DECLARE target_source  STRING DEFAULT 'historical';                          -- historical | live
DECLARE target_run     STRING DEFAULT '2022-01-01_2022-04-01_20261011-120000';
DECLARE check_trips    BOOL   DEFAULT TRUE;    -- FALSE for a live outcomes-only run
DECLARE check_outcomes BOOL   DEFAULT TRUE;    -- FALSE for a live trips-only run

-- the date window is part of the run_id: <start>_<end>_<timestamp>
DECLARE run_start DATE DEFAULT DATE(SPLIT(target_run, '_')[OFFSET(0)]);
DECLARE run_end   DATE DEFAULT DATE(SPLIT(target_run, '_')[OFFSET(1)]);   -- exclusive


-- ======================= TRIPS =======================
IF check_trips THEN
  -- copy this run's rows once; every check below reads from this small table
  CREATE TEMP TABLE run_trips AS
  SELECT * FROM `chicago-taxi-tips-project.taxi_lake.silver_trips`
  WHERE source = target_source AND run_id = target_run;

  -- summary, shown in the results
  SELECT COUNT(*) AS rows, COUNT(DISTINCT unique_key) AS unique_keys,
         MIN(DATE(trip_start_ts)) AS first_day, MAX(DATE(trip_start_ts)) AS last_day
  FROM run_trips;

  ASSERT (SELECT COUNT(*) FROM run_trips) > 0
    AS 'trips: run is empty or run_id is wrong';

  ASSERT (SELECT COUNT(*) = COUNT(DISTINCT unique_key) FROM run_trips)
    AS 'trips: duplicate unique_key';

  ASSERT (SELECT COUNTIF(unique_key IS NULL OR trip_start_ts IS NULL OR fare IS NULL
                         OR trip_seconds IS NULL OR trip_miles IS NULL) = 0 FROM run_trips)
    AS 'trips: missing required values';

  ASSERT (SELECT COUNTIF(fare <= 0 OR trip_miles <= 0 OR trip_seconds <= 0) = 0 FROM run_trips)
    AS 'trips: zero or negative fare, distance or duration';

  ASSERT (SELECT COUNTIF(DATE(trip_start_ts) < run_start
                         OR DATE(trip_start_ts) >= run_end) = 0 FROM run_trips)
    AS 'trips: trip dates outside the run window';
END IF;


-- ======================= OUTCOMES =======================
IF check_outcomes THEN
  CREATE TEMP TABLE run_outcomes AS
  SELECT * FROM `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
  WHERE source = target_source AND run_id = target_run;

  SELECT COUNT(*) AS rows, COUNT(DISTINCT unique_key) AS unique_keys,
         ROUND(AVG(tips), 2) AS avg_tip
  FROM run_outcomes;

  ASSERT (SELECT COUNT(*) FROM run_outcomes) > 0
    AS 'outcomes: run is empty or run_id is wrong';

  ASSERT (SELECT COUNT(*) = COUNT(DISTINCT unique_key) FROM run_outcomes)
    AS 'outcomes: duplicate unique_key';

  ASSERT (SELECT COUNTIF(unique_key IS NULL OR tips IS NULL OR tips < 0) = 0 FROM run_outcomes)
    AS 'outcomes: missing or negative tips';
END IF;


-- ======================= MATCH RATE (information only, no ASSERT) =======================
-- Share of this run's trips that have an outcome in ANY run of the same source.
-- Historical: ~100%. Live: lower until the outcomes have arrived.
IF check_trips THEN
  SELECT COUNT(*) AS trips,
         ROUND(COUNTIF(o.unique_key IS NOT NULL) / COUNT(*), 4) AS match_rate
  FROM run_trips t
  LEFT JOIN (SELECT DISTINCT unique_key
             FROM `chicago-taxi-tips-project.taxi_lake.silver_outcomes`
             WHERE source = target_source) o
  USING (unique_key);
END IF;
