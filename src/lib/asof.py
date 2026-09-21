"""
The no-look-ahead guard.

Why this file exists
--------------------
Every pre-death measurement in this design is frozen at the moment of death and
never updated with what happened afterwards: dominance, field size,
concentration, network position, and later the two firm coordinates. That is
not tidiness. If post-death information leaks into a pre-death measurement, the
measurement partly *contains* the outcome, and the estimate is contaminated in
the direction of the finding. The numbers stay plausible. Nothing crashes. The
paper is wrong.

The leak is easy to create by accident. Forward citations keep accruing for
decades, so "the star's citation share" computed today includes citations made
after the death. CPC codes get reclassified, so a patent can enter the star's
field years after the star has gone. An assignee's identity gets updated. None
of these announces itself.

So every measurement takes an explicit `as_of` date and passes its inputs
through the checks here first. This costs one line per measurement and catches
the whole class of error.

Usage
-----
    from src.lib import asof

    asof.check_no_future_dates(patents, death_date, ["filing_date"], label="dominance input")
    asof.check_no_future_years(panel, death_year, ["application_year"], label="field size")
    asof.check_no_future_years_by_row(windows, "filing_year", "year", label="dominance windows")

All three accept a pandas DataFrame or a DuckDB relation, and all three raise
LookAheadError naming the column, how many rows are too late, and by how much.
"""

import datetime

import duckdb
import pandas as pd


class LookAheadError(Exception):
    """Raised when data that postdates the measurement date reaches a measurement."""


def check_no_future_dates(frame, as_of, date_columns, label="", require_no_nulls=False):
    """
    Confirm that no row in `date_columns` is later than `as_of`.

    frame             a pandas DataFrame or a DuckDB relation
    as_of             a date, a datetime, or an ISO string like '1998-03-15'
    date_columns      the columns that must not postdate as_of
    label             what is being measured, so the error says which one leaked
    require_no_nulls  raise on missing dates too. Off by default: a missing date
                      is a data-quality question for the funnel, not a look-ahead
                      question. It is always reported in the summary.

    Returns a summary dictionary per column; raises LookAheadError on a leak.
    """
    cutoff = _as_date(as_of)
    summary = {}

    for column in date_columns:
        _require_column(frame, column)
        _require_date_type(frame, column)
        n_rows, n_null, n_after, latest = _date_stats(frame, column, cutoff)

        summary[column] = {"rows": n_rows, "nulls": n_null, "after_as_of": n_after,
                           "latest": latest}

        if n_after:
            raise LookAheadError(
                f"Look-ahead in {label or 'this measurement'}: {n_after:,} of {n_rows:,} rows "
                f"have {column} after the as-of date {cutoff}. The latest is {latest}.\n"
                f"A measurement frozen at {cutoff} cannot be computed from data that did not "
                f"exist yet. Filter the input to {column} <= {cutoff} before measuring — "
                f"through the funnel, so the drop is recorded."
            )
        if require_no_nulls and n_null:
            raise LookAheadError(
                f"{n_null:,} of {n_rows:,} rows have no {column} at all, so they cannot be "
                f"placed in time relative to the as-of date {cutoff}."
            )

    return summary


def check_no_future_years(frame, as_of_year, year_columns, label="", require_no_nulls=False):
    """
    The same guard for columns that hold a year rather than a full date.

    Most of this analysis runs on application *years*, not dates, so this is the
    version that gets used most. A row from the same year as `as_of_year` is
    allowed: the year is the unit, so it is not in the future.
    """
    if not isinstance(as_of_year, int) or isinstance(as_of_year, bool):
        raise TypeError(f"as_of_year should be a whole year like 1998, not {as_of_year!r}.")

    summary = {}
    for column in year_columns:
        _require_column(frame, column)
        n_rows, n_null, n_after, latest = _year_stats(frame, column, as_of_year)

        summary[column] = {"rows": n_rows, "nulls": n_null, "after_as_of": n_after,
                           "latest": latest}

        if n_after:
            raise LookAheadError(
                f"Look-ahead in {label or 'this measurement'}: {n_after:,} of {n_rows:,} rows "
                f"have {column} after {as_of_year}. The latest is {latest}.\n"
                f"Filter the input to {column} <= {as_of_year} before measuring — through the "
                f"funnel, so the drop is recorded."
            )
        if require_no_nulls and n_null:
            raise LookAheadError(
                f"{n_null:,} of {n_rows:,} rows have no {column}, so they cannot be placed in "
                f"time relative to {as_of_year}."
            )

    return summary


def check_no_future_years_by_row(frame, year_column, as_of_column, label=""):
    """
    The same guard when every row carries its own as-of year.

    The two functions above check a whole table against ONE date, which is the
    right shape for one event: everything measured for a star who died in 1998
    must predate 1998. But a rolling window table holds every measurement year
    at once — an inventor's five-year window ending 1994 and their window ending
    2003 sit in the same file, each with its own deadline. Checking that table
    against a single date is either vacuous or wrong.

    So this compares two columns row by row: the year the data comes from
    (`year_column`) against the year the measurement is frozen at
    (`as_of_column`). A row is legal when they are equal — the year is the unit,
    so a patent filed in the window's closing year is not in the future.

    One pass over the table checks every measurement year simultaneously, which
    is both faster and stronger than looping check_no_future_years over the
    years: nothing depends on remembering to loop over all of them.

    Returns a summary dictionary; raises LookAheadError on a leak.
    """
    _require_column(frame, year_column)
    _require_column(frame, as_of_column)

    n_rows, n_after, worst = _row_pair_stats(frame, year_column, as_of_column)
    summary = {"rows": n_rows, "after_as_of": n_after, "worst_years_ahead": worst}

    if n_after:
        raise LookAheadError(
            f"Look-ahead in {label or 'this measurement'}: {n_after:,} of {n_rows:,} rows have "
            f"{year_column} after their own {as_of_column}, the worst by {worst} years.\n"
            f"A window that closes in {as_of_column} cannot contain data filed afterwards. "
            f"Build the window forward from the source year instead of joining a year range, "
            f"so a later row cannot reach an earlier window at all."
        )
    return summary


# ---------------------------------------------------------------------------
# Small helpers. Two branches everywhere — pandas and DuckDB — on purpose:
# a shared abstraction over the two would be harder to read than the repetition.
# ---------------------------------------------------------------------------

def _as_date(value):
    """Accept a date, a datetime or an ISO string; refuse anything else."""
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            raise TypeError(
                f"as_of should look like '1998-03-15', not {value!r}."
            ) from None
    raise TypeError(f"as_of should be a date or an ISO date string, not {type(value).__name__}.")


def _columns_of(frame):
    if isinstance(frame, pd.DataFrame):
        return list(frame.columns)
    if isinstance(frame, duckdb.DuckDBPyRelation):
        return list(frame.columns)
    raise TypeError(
        f"Expected a pandas DataFrame or a DuckDB relation, got {type(frame).__name__}."
    )


def _require_column(frame, column):
    columns = _columns_of(frame)
    if column not in columns:
        raise KeyError(
            f"There is no column '{column}' to check. Available: {', '.join(columns)}."
        )


def _require_date_type(frame, column):
    """
    Refuse to compare dates that are still text.

    This looks pedantic and is not. Comparing '1998-03-15' as a string happens to
    work; comparing '3/15/1998' as a string silently does not, and PatentsView
    contains both real dates and impossible ones like '1074-08-14'. A guard that
    quietly compares the wrong thing is worse than no guard.
    """
    if isinstance(frame, pd.DataFrame):
        if not pd.api.types.is_datetime64_any_dtype(frame[column]):
            raise TypeError(
                f"Column '{column}' is {frame[column].dtype}, not a date. Convert it to a "
                f"real date first — where a failed conversion is visible and recorded — "
                f"rather than comparing dates as text."
            )
        return
    kind = str(frame.types[frame.columns.index(column)]).upper()
    if not any(k in kind for k in ("DATE", "TIMESTAMP")):
        raise TypeError(
            f"Column '{column}' is {kind}, not a date. Step 01 loads everything as text on "
            f"purpose; convert it to a real date in the step, where a failed conversion is "
            f"visible and recorded, rather than comparing dates as text."
        )


def _date_stats(frame, column, cutoff):
    """(rows, nulls, rows after cutoff, latest value) for a date column."""
    if isinstance(frame, pd.DataFrame):
        series = frame[column]
        after = series > pd.Timestamp(cutoff)
        latest = series.max()
        return len(series), int(series.isna().sum()), int(after.sum()), latest

    n_rows = frame.aggregate("count(*) AS n").fetchone()[0]
    n_null = frame.filter(f'"{column}" IS NULL').aggregate("count(*) AS n").fetchone()[0]
    n_after = frame.filter(f"\"{column}\" > DATE '{cutoff}'").aggregate(
        "count(*) AS n").fetchone()[0]
    latest = frame.aggregate(f'max("{column}") AS m').fetchone()[0]
    return n_rows, n_null, n_after, latest


def _row_pair_stats(frame, year_column, as_of_column):
    """(rows, rows where year > as_of, worst overshoot in years) for two columns."""
    if isinstance(frame, pd.DataFrame):
        ahead = frame[year_column] - frame[as_of_column]
        late = ahead > 0
        return len(frame), int(late.sum()), int(ahead.max()) if len(frame) else 0

    n_rows = frame.aggregate("count(*) AS n").fetchone()[0]
    n_after = frame.filter(f'"{year_column}" > "{as_of_column}"').aggregate(
        "count(*) AS n").fetchone()[0]
    worst = frame.aggregate(f'max("{year_column}" - "{as_of_column}") AS m').fetchone()[0]
    return n_rows, n_after, worst


def _year_stats(frame, column, as_of_year):
    """(rows, nulls, rows after the year, latest year) for a year column."""
    if isinstance(frame, pd.DataFrame):
        series = frame[column]
        return (len(series), int(series.isna().sum()), int((series > as_of_year).sum()),
                series.max())

    n_rows = frame.aggregate("count(*) AS n").fetchone()[0]
    n_null = frame.filter(f'"{column}" IS NULL').aggregate("count(*) AS n").fetchone()[0]
    n_after = frame.filter(f'"{column}" > {as_of_year}').aggregate("count(*) AS n").fetchone()[0]
    latest = frame.aggregate(f'max("{column}") AS m').fetchone()[0]
    return n_rows, n_null, n_after, latest
