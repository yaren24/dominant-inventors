"""
Tests for the funnel logger.

What these protect: the funnel is the methods section. If a filter can run
without an explanation, or the counts in the diary do not match what the code
did, the thesis cannot defend its own sample.
"""

import csv

import duckdb
import pandas as pd
import pytest

from src.lib.funnel import Funnel


WHY = ("Design and plant patents are legally patents but are not inventions in the sense "
       "this design is about, so they are removed before any share is computed.")


def patents():
    """Four patents, three inventors, two of them on the same patent."""
    return pd.DataFrame({
        "patent_id": ["P1", "P1", "P2", "P3", "P4"],
        "inventor_id": ["I1", "I2", "I1", "I3", "I3"],
        "patent_type": ["utility", "utility", "utility", "design", "plant"],
        "application_year": [1994, 1994, 1996, 1997, 1999],
    })


def test_a_filter_records_rows_patents_and_inventors(logging_config):
    funnel = Funnel("test_step", logging_config)
    kept = funnel.filter(patents(), "utility_only", "patent_type = 'utility'", why=WHY)

    assert len(kept) == 3
    entry = funnel.steps[0]
    assert entry["rows_before"] == 5 and entry["rows_after"] == 3
    assert entry["patents_before"] == 4 and entry["patents_after"] == 2
    assert entry["inventors_before"] == 3 and entry["inventors_after"] == 2


def test_a_filter_without_a_real_explanation_is_refused(logging_config):
    """The why-sentence is not optional and 'cleanup' is not a sentence."""
    funnel = Funnel("test_step", logging_config)
    with pytest.raises(ValueError, match="methods section"):
        funnel.filter(patents(), "utility_only", "patent_type = 'utility'", why="cleanup")


def test_the_same_sql_works_on_a_duckdb_relation(logging_config):
    """Real tables are far too large for pandas, so both kinds must behave identically."""
    con = duckdb.connect()
    con.register("patents", patents())
    relation = con.sql("SELECT * FROM patents")

    funnel = Funnel("test_step", logging_config)
    kept = funnel.filter(relation, "utility_only", "patent_type = 'utility'", why=WHY)

    assert kept.aggregate("count(*) AS n").fetchone()[0] == 3
    assert funnel.steps[0]["patents_after"] == 2


def test_counts_are_left_blank_when_the_column_is_not_in_the_table(logging_config):
    """A field-year table has no inventor column; that must read as blank, not as zero."""
    fields = pd.DataFrame({"field_id": ["A61K", "A61B"], "patents": [120, 40]})
    funnel = Funnel("test_step", logging_config)
    funnel.filter(fields, "big_enough", "patents >= 100",
                  why="Fields with very few patents are excluded because a large share of a "
                      "tiny field is not dominance of anything.")
    entry = funnel.steps[0]
    assert entry["inventors_before"] is None
    assert entry["rows_before"] == 2 and entry["rows_after"] == 1


def test_checkpoints_and_notes_are_recorded_without_dropping_anything(logging_config):
    funnel = Funnel("test_step", logging_config)
    funnel.checkpoint("after loading", patents())
    funnel.note("Two patents share an inventor; that is expected here.")

    kinds = [entry["kind"] for entry in funnel.steps]
    assert kinds == ["checkpoint", "note"]
    assert funnel.steps[0]["rows_before"] == funnel.steps[0]["rows_after"] == 5


def test_finishing_writes_all_four_files(logging_config):
    funnel = Funnel("test_step", logging_config)
    funnel.filter(patents(), "utility_only", "patent_type = 'utility'", why=WHY)
    run_dir = funnel.finish(print_to_terminal=False)

    for name in ("funnel.csv", "data_diary.md", "config_used.yaml", "session.log"):
        assert (run_dir / name).exists(), f"{name} was not written"


def test_the_diary_contains_the_why_sentence_and_the_counts(logging_config):
    funnel = Funnel("test_step", logging_config)
    funnel.filter(patents(), "utility_only", "patent_type = 'utility'", why=WHY)
    run_dir = funnel.finish(print_to_terminal=False)

    diary = (run_dir / "data_diary.md").read_text(encoding="utf-8")
    assert WHY in diary                      # the sentence, verbatim
    assert "5 → 3" in diary                  # the rows
    assert logging_config.hash in diary      # which config produced it


def test_the_csv_has_one_row_per_step_with_the_drop_share(logging_config):
    funnel = Funnel("test_step", logging_config)
    funnel.filter(patents(), "utility_only", "patent_type = 'utility'", why=WHY)
    funnel.filter(patents(), "early_only", "application_year <= 1996",
                  why="Applications after 1996 are outside the measurement window used here, "
                      "so they cannot contribute to a share measured over 1992 to 1996.")
    run_dir = funnel.finish(print_to_terminal=False)

    with open(run_dir / "funnel.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["name"] == "utility_only"
    assert rows[0]["rows_dropped"] == "2"
    assert rows[0]["share_dropped"] == "0.4"
    assert rows[0]["why"] == WHY


def test_the_config_snapshot_is_the_file_byte_for_byte(logging_config):
    funnel = Funnel("test_step", logging_config)
    run_dir = funnel.finish(print_to_terminal=False)
    saved = (run_dir / "config_used.yaml").read_text(encoding="utf-8")
    assert saved == logging_config.source_path.read_text(encoding="utf-8")


def test_the_session_log_is_written_as_the_run_proceeds(logging_config):
    """A crash halfway through must still leave a trail of what had happened."""
    funnel = Funnel("test_step", logging_config)
    funnel.filter(patents(), "utility_only", "patent_type = 'utility'", why=WHY)

    log = (funnel.run_dir / "session.log").read_text(encoding="utf-8")
    assert "run started" in log
    assert "utility_only" in log             # present before finish() was ever called
