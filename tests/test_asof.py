"""
Tests for the no-look-ahead guard.

The important test in this file is the one that deliberately feeds a row dated
after the measurement date and confirms the guard fires. A look-ahead leak does
not crash and does not look wrong — it produces a plausible number that is
contaminated by the outcome. If this test ever stops failing on bad input, the
protection has quietly gone.
"""

import datetime

import duckdb
import pandas as pd
import pytest

from src.lib import asof


def patents_frame(dates):
    """A tiny patent table with real date types, as the guard requires."""
    return pd.DataFrame({
        "patent_id": [f"P{i}" for i in range(len(dates))],
        "filing_date": pd.to_datetime(dates),
    })


DEATH = datetime.date(1998, 6, 30)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def test_data_entirely_before_the_death_passes():
    frame = patents_frame(["1994-01-01", "1996-07-15", "1998-06-29"])
    summary = asof.check_no_future_dates(frame, DEATH, ["filing_date"], label="dominance")
    assert summary["filing_date"]["after_as_of"] == 0
    assert summary["filing_date"]["rows"] == 3


def test_a_row_dated_after_the_death_raises():
    """THE test. One patent filed the day after the death must stop the run."""
    frame = patents_frame(["1994-01-01", "1998-07-01"])

    with pytest.raises(asof.LookAheadError) as raised:
        asof.check_no_future_dates(frame, DEATH, ["filing_date"], label="dominance share")

    message = str(raised.value)
    assert "1 of 2 rows" in message
    assert "filing_date" in message
    assert "1998-06-30" in message
    assert "dominance share" in message      # says WHICH measurement leaked


def test_a_row_on_the_death_date_itself_passes():
    """The cutoff is inclusive: what existed on the day is pre-death information."""
    frame = patents_frame(["1998-06-30"])
    asof.check_no_future_dates(frame, DEATH, ["filing_date"])


def test_the_same_guard_works_on_a_duckdb_relation():
    """Most of the real data never becomes a DataFrame, so the guard must work on both."""
    con = duckdb.connect()
    relation = con.sql("SELECT 'P1' AS patent_id, DATE '1999-01-01' AS filing_date")
    with pytest.raises(asof.LookAheadError, match="after the as-of date"):
        asof.check_no_future_dates(relation, DEATH, ["filing_date"])


def test_dates_still_stored_as_text_are_refused():
    """
    Step 01 loads everything as text on purpose, and PatentsView contains
    impossible dates like '1074-08-14'. Comparing those as strings sometimes
    works and sometimes silently does not, which is worse than not checking.
    """
    frame = pd.DataFrame({"filing_date": ["1998-07-01", "1074-08-14"]})
    with pytest.raises(TypeError, match="not a date"):
        asof.check_no_future_dates(frame, DEATH, ["filing_date"])


def test_missing_dates_are_reported_but_not_fatal_by_default():
    frame = patents_frame(["1994-01-01", None])
    summary = asof.check_no_future_dates(frame, DEATH, ["filing_date"])
    assert summary["filing_date"]["nulls"] == 1


def test_missing_dates_can_be_made_fatal():
    frame = patents_frame(["1994-01-01", None])
    with pytest.raises(asof.LookAheadError, match="cannot be placed in time"):
        asof.check_no_future_dates(frame, DEATH, ["filing_date"], require_no_nulls=True)


def test_a_column_that_does_not_exist_says_which_ones_do():
    frame = patents_frame(["1994-01-01"])
    with pytest.raises(KeyError, match="filing_date"):
        asof.check_no_future_dates(frame, DEATH, ["application_date"])


def test_the_as_of_date_can_be_a_string_but_not_nonsense():
    frame = patents_frame(["1994-01-01"])
    asof.check_no_future_dates(frame, "1998-06-30", ["filing_date"])
    with pytest.raises(TypeError, match="30 June 1998"):
        asof.check_no_future_dates(frame, "30 June 1998", ["filing_date"])


# ---------------------------------------------------------------------------
# Years — the version this analysis actually uses most, since everything is
# dated by application year.
# ---------------------------------------------------------------------------

def years_frame(years):
    return pd.DataFrame({
        "inventor_id": ["A"] * len(years),
        "application_year": years,
    })


def test_years_up_to_and_including_the_death_year_pass():
    frame = years_frame([1994, 1996, 1998])
    summary = asof.check_no_future_years(frame, 1998, ["application_year"])
    assert summary["application_year"]["after_as_of"] == 0


def test_a_later_application_year_raises():
    frame = years_frame([1994, 1999])
    with pytest.raises(asof.LookAheadError, match="after 1998"):
        asof.check_no_future_years(frame, 1998, ["application_year"], label="field size")


def test_the_year_guard_works_on_a_duckdb_relation():
    con = duckdb.connect()
    relation = con.sql("SELECT 2001 AS application_year")
    with pytest.raises(asof.LookAheadError):
        asof.check_no_future_years(relation, 1998, ["application_year"])


def test_a_date_passed_where_a_year_is_expected_is_refused():
    frame = years_frame([1994])
    with pytest.raises(TypeError, match="whole year"):
        asof.check_no_future_years(frame, datetime.date(1998, 6, 30), ["application_year"])


# ---------------------------------------------------------------------------
# Rolling windows — where every row carries its own deadline.
#
# This is the shape Task 2 produces: an inventor's five-year window closing in
# 1994 and their window closing in 2003 sit in the same table. Checking that
# table against one date would be either vacuous or wrong, so the guard compares
# two columns instead.
# ---------------------------------------------------------------------------

def windows_frame(pairs):
    """(filing_year, window_year) pairs — the year data came from, and its deadline."""
    return pd.DataFrame({
        "inventor_id": ["A"] * len(pairs),
        "filing_year": [p[0] for p in pairs],
        "year": [p[1] for p in pairs],
    })


def test_a_backward_looking_window_passes():
    # A patent filed in 1990 legitimately enters the windows closing 1990-1994.
    frame = windows_frame([(1990, 1990), (1990, 1994), (2001, 2003)])
    summary = asof.check_no_future_years_by_row(frame, "filing_year", "year")
    assert summary["after_as_of"] == 0
    assert summary["rows"] == 3


def test_a_patent_reaching_backwards_into_an_earlier_window_raises():
    # The failure this whole module exists to prevent: a 1996 patent counted in
    # the window that closed in 1994, which would let the future decide who was
    # dominant before it happened.
    frame = windows_frame([(1990, 1994), (1996, 1994)])
    with pytest.raises(asof.LookAheadError, match="worst by 2 years"):
        asof.check_no_future_years_by_row(frame, "filing_year", "year",
                                          label="dominance windows")


def test_the_row_wise_guard_works_on_a_duckdb_relation():
    con = duckdb.connect()
    relation = con.sql("SELECT 2001 AS filing_year, 1998 AS year")
    with pytest.raises(asof.LookAheadError, match="dominance windows"):
        asof.check_no_future_years_by_row(relation, "filing_year", "year",
                                          label="dominance windows")


def test_the_row_wise_guard_names_a_missing_column():
    frame = windows_frame([(1990, 1994)])
    with pytest.raises(KeyError, match="window_year"):
        asof.check_no_future_years_by_row(frame, "filing_year", "window_year")
