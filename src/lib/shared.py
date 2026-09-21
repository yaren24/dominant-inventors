"""
Helpers shared by more than one step.

Nothing here answers a question. These are the things more than one step needed:
where the Task 2 tables live, what the tiny-field floor is in SQL, and how to
print a run of years.

A helper used by ONE step stays in that step's file. Two lines of duplication
beat an indirection to trace, so this file only grows when a second step reaches
for the same thing.
"""

from src.lib import io


def field_years_path(cfg, width):
    return io.interim_path(cfg, f"02_field_years_{width}.parquet")


def inventor_field_years_path(cfg, width):
    return io.interim_path(cfg, f"02_inventor_field_years_{width}.parquet")


def floor_clause(cfg):
    """
    The SQL a field-year must satisfy to clear the tiny-field floor.

    Owning 30% of a field with ten patents is not dominance of anything, so
    Task 3 excludes thin fields. Since the advisor's floor change (2026-09-15)
    the floor is computed ONCE, in step 03, and stored on every field-year as
    `floor_patents` (the number) and `clears_floor` (the test) — because under
    `dominance.floor_mode: percentile` the number is a property of the
    distribution and cannot be written as a row test at all. So this clause is
    the stored flag, and it can only be evaluated on a table that carries it:
    02_field_years_<width>.parquet, or anything that joined the flag from it.
    `cfg` is kept as the argument so that no caller changes shape.
    """
    return "clears_floor"



def stored_floor(con, cfg, width):
    """
    What the floor came to, in patents, read off the field-year table.

    (lowest, highest) over the window-years inside the death window: one number
    under `absolute` or `pooled`, a range under `per_year`. Read from the file and
    never from the config, so a note describes the floor that was applied.
    """
    return con.execute(f"""
        SELECT min(floor_patents), max(floor_patents)
        FROM read_parquet('{field_years_path(cfg, width).as_posix()}')
        WHERE year BETWEEN {cfg.deaths.year_min} AND {cfg.deaths.year_max}
    """).fetchone()


def floor_words(cfg, lowest, highest):
    """`the 100-patent floor`, or `the floor of 7 patents (25th percentile, pooled)`."""
    if cfg.dominance.floor_mode == "absolute":
        return f"the {lowest:g}-patent floor"
    spec = f"{cfg.dominance.floor_percentile:g}th percentile, {cfg.dominance.floor_scope}"
    if lowest == highest:
        return f"the floor of {lowest:g} patents ({spec})"
    return f"the floor of {lowest:g} to {highest:g} patents by year ({spec})"


def compress_years(years):
    """
    [1994, 1995, 1996, 1997, 1998, 2013] -> '1994-1998, 2013'.

    Only for reading. The complete year-by-year list is in the CSV; a table of
    thirty subclasses each listing eight separate years is a table nobody reads.
    """
    runs = []
    for year in sorted(years):
        if runs and year == runs[-1][1] + 1:
            runs[-1][1] = year
        else:
            runs.append([year, year])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
