"""
Step 03 — Task 2: measure dominance for EVERY inventor, at three field widths.

Reads : data/interim/01_inventor_patents.parquet   (the Task 1 spine)
        data/interim/01_patents.parquet            (for field size head counts)
        data/interim/g_cpc_*.tsv.parquet           (through src/lib/fields.py)
Writes: data/interim/02_inventor_field_years_<width>.parquet
        data/interim/02_field_years_<width>.parquet   (carries the field floor:
                                                       floor_patents, clears_floor)
        outputs/<config_name>/task2_validation.md
        logs/runs/<timestamp>_03_measure_dominance/

Config keys used: task2.widths, windows.dominance_window_years,
dominance.first_compute_year, dominance.last_compute_year,
dominance.rank_tie_rule, dominance.weight_rounding_decimals,
dominance.margin_when_no_runner_up, dominance.margin_cap (read only under
margin_when_no_runner_up: cap, and then as a fill value, not a ceiling — see
below), dominance.compute_citation_share, dominance.floor_mode,
dominance.min_field_patents, dominance.min_field_patents_inclusive,
dominance.floor_percentile, dominance.floor_scope (the last two only under
floor_mode: percentile), counting.method, cpc.assignment,
cpc.multi_code_weighting, counting.field_size_method, fields.strategy,
patent_universe.first_year, deaths.year_min, deaths.year_max, sample_mode,
sample_cpc_section, runtime.*

Task in the task list: Task 2, the main build. **Deaths are ignored entirely.**
This step does not know that anybody dies.

Serves: SQ1 and SQ5. Dominance is the continuous treatment DOSE in both, not
merely a sample filter, which is why nothing here is discretised.

For every inventor x field x year this computes, over the window t-4 through t:

  share    the inventor's share of the field's patents filed in the window
  rank     the inventor's rank in the field over the same window
  margin   for the rank-1 inventor, the #1 share divided by the #2 share —
           stored twice, under two definitions of "#2": the second ROW of the
           ordering (margin_uncapped, a tied top reads 1.0) and the second
           DISTINCT weight (margin_dense, tied co-leaders count as one leader).
           See dominance_margin().

And for every field-year, the field floor: floor_patents, the number of window
patents a field-year needs, and clears_floor, whether this one has it. Computed
ONCE here and read everywhere else, so the count family, the citation family and
Task 4 cannot floor at different numbers. Under dominance.floor_mode: absolute
the number is dominance.min_field_patents; under percentile it is a quantile of
the field-year size distribution at this width (see field_floor_sql()). This
step FLAGS the floor and applies it nowhere: every field-year is still written.

The three rules the advisor is explicit about, and where each one lives:

  1. Numerator and denominator use the same counting. Enforced structurally in
     add_share_rank_and_position() — the denominator is a window function OVER
     THE NUMERATOR COLUMN, so there is no second column that could be
     substituted. Backed by the sum-to-one check in validate().
  2. The margin is stored UNCAPPED: no ratio computed here is ever clipped to
     dominance.margin_cap, which belongs to the figures in step 02b. This file
     does read that key, in exactly one place — as the value written when
     margin_when_no_runner_up is `cap` and a field-year has no #2 at all. Filling
     a margin that does not exist is not the same act as capping one that does,
     and the earlier wording of this line said the key was never read here (C-42).
  3. The window is backward-looking. Built forward from the filing year in
     window_rows_sql(), so a patent cannot reach an earlier window at all, and
     verified row by row through src/lib/asof.py.

What this step does NOT do: no dominance cutoff, no tiny-field exclusion (the
floor is flagged, not applied — no row is dropped for it), no margin cap, no
citation share. Those are Task 3, Task 3, the figures, and Task 6 respectively.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb

from src.lib import asof
from src.lib import config as config_module
from src.lib import fields as fields_module
from src.lib import io
from src.lib.funnel import Funnel


# How a rank is formed when two inventors hold exactly the same window weight.
# The rule is chosen in the config; this is only the translation into SQL, so
# that the three options cannot drift apart from the three implementations.
# Note that `unique_by_id` orders by inventor_id as well, so that it is
# reproducible: a tie broken by evaluation order would make two runs of the same
# config disagree, which is the bug Task 1 hit with arg_max.
RANK_EXPRESSIONS = {
    "competition": "rank() OVER (PARTITION BY field_id, year "
                   "ORDER BY inventor_weight_in_window DESC)",
    "dense": "dense_rank() OVER (PARTITION BY field_id, year "
             "ORDER BY inventor_weight_in_window DESC)",
    "unique_by_id": "row_number() OVER (PARTITION BY field_id, year "
                    "ORDER BY inventor_weight_in_window DESC, inventor_id)",
}


def interim(cfg, name):
    return io.interim_path(cfg, name)


def inventor_file(cfg, width):
    return interim(cfg, f"02_inventor_field_years_{width}.parquet")


def field_file(cfg, width):
    return interim(cfg, f"02_field_years_{width}.parquet")


# ---------------------------------------------------------------------------
# 1. The spine at one field width, with the counting weight
# ---------------------------------------------------------------------------

def counting_weight_sql(cfg):
    """
    The expression that turns a patent into a person's credit.

    This is written out here AND in src/02_build_patent_tables.py. That duplication is
    deliberate but dangerous: if the two ever disagree, every share in this
    project would have a numerator and a denominator built on different
    definitions, and the numbers would stay entirely plausible. So the two are
    not left to trust — check_matches_task_one_spine() below rebuilds the spine
    at the config's own field width and refuses to continue unless it reproduces
    Task 1's file row for row, weight for weight.

    The expression is only valid inside build_spine_at_width()'s query, because
    it names that query's columns: `team_size`, `fields_on_patent` and, under full
    counting, `appearances`.
    """
    # Under FULL counting the divisor is `appearances`, not 1, and the two cancel
    # in build_spine_at_width below to exactly 1.0 per patent per person. That is
    # what "a patent counts 1 for each inventor" means: a person the office listed
    # three times on one patent is one inventor with one patent, not three. The
    # mirror of this line in src/02_build_patent_tables.py divides by `listings`, which is
    # the same count taken before the rows are collapsed. See notes/decisions.md
    # C-38 — this branch had never been run, and it credited 857 people with
    # patents they do not hold.
    per_inventor = ("1.0 / team_size" if cfg.counting.method == "fractional"
                    else "1.0 / appearances")

    # Under the all-codes alternative a patent belongs to several fields at once,
    # and cpc.multi_code_weighting decides whether it counts fully in each or is
    # split across them. Under the primary-code default there is only ever one
    # field, so the divisor is 1 and this line does nothing.
    per_field = ("fields_on_patent"
                 if (cfg.cpc.assignment == "all"
                     and cfg.cpc.multi_code_weighting == "fractional")
                 else "1")
    return f"({per_inventor}) / {per_field}"


def build_spine_at_width(con, cfg, width, funnel):
    """
    One row per inventor x patent x field, at this width, carrying the weight.

    The field assignment is redone here rather than taken from the Task 1 spine,
    because Task 1 froze one width into its field_id and because under
    cpc.assignment: all the counting weight itself depends on how many fields a
    patent has. Inheriting either would silently give the wrong answer at two of
    the three widths.
    """
    fields_module.assign_fields(con, cfg, funnel=funnel, strategy=width).to_view(
        "patent_fields_at_width", replace=True)

    # Collapse the Task 1 spine's field dimension, which is fixed at the wrong
    # width, while keeping everything that is width-independent: the patent, the
    # person, the team size and the year.
    #
    # `appearances` is not pedantry. 1,779 patent-inventor pairs in this data list
    # the SAME disambiguated inventor two or three times on one patent — patent
    # 6135931 names fl:wi_ln:padula-1 three times and has a team size of three,
    # so the whole patent is one person recorded thrice. Task 1 gives that person
    # the entire patent, which is right. Counting the pair once instead would
    # hand them a third of it and let the other two thirds belong to nobody,
    # quietly shrinking the field denominator. The inner DISTINCT is what makes
    # this safe under cpc.assignment: all, where each appearance repeats once per
    # field.
    #
    # That argument is about FRACTIONAL counting, and it does not carry over.
    # Under full counting the patent is worth 1 to each inventor, so multiplying
    # by `appearances` credited the padula case with three patents — which is why
    # counting_weight_sql() divides by `appearances` there and the two cancel.
    # 857 people were affected across 5,621 inventor-field-year rows, 49 of them
    # rank-1 rows. Measured and fixed at Task 2b item 9; see C-38.
    #
    # The 1,779 was 381 when this comment was written, and moved with Task 1's
    # three rebuilds (C-21, C-23, C-25). Numbers in comments go stale silently,
    # which is the reason tests/test_task2b_item9.py drives this function rather
    # than trusting the paragraph.
    con.sql(f"""
        SELECT patent_id, inventor_id, team_size, year, count(*) AS appearances
        FROM (
            SELECT DISTINCT patent_id, inventor_id, inventor_sequence, team_size, year
            FROM read_parquet('{interim(cfg, "01_inventor_patents.parquet").as_posix()}')
        )
        GROUP BY 1, 2, 3, 4
    """).to_view("patent_inventors", replace=True)

    con.sql("""
        SELECT patent_id, field_id,
               count(*) OVER (PARTITION BY patent_id) AS fields_on_patent
        FROM patent_fields_at_width
    """).to_view("fields_with_count", replace=True)

    spine = con.sql(f"""
        SELECT
            i.patent_id,
            i.inventor_id,
            f.field_id,
            i.year                                       AS filing_year,
            i.appearances * ({counting_weight_sql(cfg)}) AS weight
        FROM patent_inventors i
        JOIN fields_with_count f ON f.patent_id = i.patent_id
    """)
    funnel.checkpoint(f"{width}: inventor x patent x field", spine)
    return spine


def check_matches_task_one_spine(con, cfg, width):
    """
    Refuse to continue unless this step reproduces Task 1's spine exactly.

    Only meaningful at the width Task 1 was built with. It compares every row —
    patent, inventor, field and weight — in both directions, so a weight
    expression that has drifted between the two files cannot survive, whichever
    way it drifted.

    Both sides are aggregated to one row per patent x inventor x field first.
    That is not tidying: EXCEPT is set-based, so comparing raw rows would treat
    an inventor listed twice on a patent as one row and report two files as
    identical while one of them was missing weight. That is exactly what
    happened on the first run of this step.
    """
    if width != cfg.fields.strategy:
        return f"not checked at this width (Task 1 was built at {cfg.fields.strategy})"

    decimals = cfg.dominance.weight_rounding_decimals
    mismatched = con.sql(f"""
        WITH here AS (
            SELECT patent_id, inventor_id, field_id, round(sum(weight), {decimals}) AS weight
            FROM spine_at_width GROUP BY 1, 2, 3
        ),
        task1 AS (
            SELECT patent_id, inventor_id, field_id, round(sum(weight), {decimals}) AS weight
            FROM read_parquet('{interim(cfg, "01_inventor_patents.parquet").as_posix()}')
            GROUP BY 1, 2, 3
        )
        SELECT (SELECT count(*) FROM (SELECT * FROM here EXCEPT SELECT * FROM task1))
             + (SELECT count(*) FROM (SELECT * FROM task1 EXCEPT SELECT * FROM here))
    """).fetchone()[0]

    if mismatched:
        raise ValueError(
            f"{mismatched:,} rows differ between this step's spine and Task 1's "
            f"01_inventor_patents.parquet at width {width}. The counting weight in "
            f"src/03_measure_dominance.py has drifted from the one in src/02_build_patent_tables.py. "
            f"Every share computed from here would have a numerator and a denominator built "
            f"on different definitions. Fix counting_weight_sql() before going further."
        )
    return "identical to Task 1, row for row"


# ---------------------------------------------------------------------------
# 2. The backward-looking window
# ---------------------------------------------------------------------------

def window_rows_sql(cfg):
    """
    The ONE place the five-year window is constructed.

    A patent filed in year f is emitted at window-years f, f+1 ... f+4. That is
    the same thing as "window t covers filings t-4 through t", built the other
    way round — and building it forward is what makes the no-look-ahead property
    structural rather than a promise. There is no year range to join on, so
    there is no join condition that could be written with the inequality
    the wrong way round.

    Returned as a string because two things need exactly these rows: the
    look-ahead guard, which reads them one by one, and the aggregation, which
    sums them. Writing the expansion twice would let the guard drift away from
    the thing it is guarding.
    """
    span = cfg.windows.dominance_window_years
    return f"""
        SELECT
            a.inventor_id,
            a.field_id,
            a.filing_year,
            a.weight_in_year,
            a.patents_in_year,
            a.filing_year + step.years_ahead AS year
        FROM activity a,
             (SELECT unnest(range(0, {span})) AS years_ahead) step
    """


def build_activity(con, cfg, funnel, width):
    """
    One row per inventor x field x FILING year — what actually happened, before
    any window is laid over it.

    Materialised rather than left lazy because the window expansion below reads
    it five times over, and because the funnel counts rows on the way past.
    """
    con.execute("""
        CREATE OR REPLACE TEMP TABLE activity AS
        SELECT inventor_id, field_id, filing_year,
               sum(weight) AS weight_in_year,
               count(*)    AS patents_in_year
        FROM spine_at_width
        GROUP BY 1, 2, 3
    """)
    funnel.checkpoint(f"{width}: inventor x field x filing year",
                      con.sql("SELECT * FROM activity"))


def build_windows(con, cfg, funnel, width):
    """
    Roll the filing years up into five-year windows, and check nothing came back
    from the future while doing it.
    """
    # Rule 3, verified rather than asserted. Every row knows which filing year it
    # came from and which window year it is entering; the guard compares the two
    # for all 40-odd window years in one pass.
    asof.check_no_future_years_by_row(
        con.sql(window_rows_sql(cfg)), "filing_year", "year",
        label=f"Task 2 dominance windows at {width}",
    )

    decimals = cfg.dominance.weight_rounding_decimals
    span = cfg.windows.dominance_window_years

    # Rounding is not cosmetic. Two weights that are mathematically equal can
    # differ in the last bit of floating point: an inventor credited
    # 1/2 + 1/3 + 1/6 and one credited 1/2 + 1/2 both hold exactly one patent's
    # worth, and those two sums are NOT equal as doubles. Unrounded, that would
    # turn a real tie at the top of a field into a strict ordering and quietly
    # hand one of two equal inventors the rank-1 row.
    windows = con.sql(f"""
        SELECT
            inventor_id,
            field_id,
            year,
            {span}                                    AS window_length_years,
            year - {span} + 1                         AS window_first_year,
            round(sum(weight_in_year), {decimals})    AS inventor_weight_in_window,
            sum(patents_in_year)                      AS inventor_patents_in_window
        FROM ({window_rows_sql(cfg)})
        GROUP BY 1, 2, 3
    """)

    windows = funnel.filter(
        windows, f"{width}_years_to_compute",
        f"year BETWEEN {cfg.dominance.first_compute_year} "
        f"AND {cfg.dominance.last_compute_year}",
        why=(
            f"Dominance is measured for window-years {cfg.dominance.first_compute_year} to "
            f"{cfg.dominance.last_compute_year} only. Earlier window-years would be built from "
            f"fewer than {span} years of data, because PatentsView begins in "
            f"{cfg.patent_universe.first_year}, and a share measured over one year is not the "
            f"same statistic as a share measured over {span} — it is far noisier and mechanically "
            f"larger. Later window-years would close on filing years that are still incomplete. "
            f"Nothing is lost for this design: the earliest usable death is "
            f"{cfg.deaths.year_min}."
        ),
    )
    return windows


# ---------------------------------------------------------------------------
# 3. Share, rank and margin
# ---------------------------------------------------------------------------

def add_share_rank_and_position(cfg):
    """
    Share, rank and sort position for every inventor-field-year.

    RULE 1 LIVES ON THE `share` LINE. The denominator is a window function over
    the numerator column itself:

        inventor_weight_in_window / sum(inventor_weight_in_window) OVER (...)

    That makes the wrong denominator harder to write, and this docstring used to
    claim it made it impossible: "there is no other column in scope that could be
    put there by mistake." **That was false, and Task 2b item 8 measured it.**
    `windows_t` also carries `inventor_patents_in_window` — one word different from
    the numerator's name — and the sum of it over the field-year is exactly the
    full-count denominator. Swapping the two is a one-word edit that type-checks,
    and under it the [0, 1] range check and the Task 1 weight reconciliation both
    still pass; only the sum-to-one check in validate() below notices. So the
    structure narrows the failure modes and does not close them, and the check is
    load-bearing rather than a formality. See tests/test_shares_sum_to_one.py,
    which performs that exact mutation, and notes/decisions.md C-37.

    The other quiet substitution is the field's patent head count. Under the
    default settings it very nearly equals the field's weight total — equal in
    74.8% of field-years at cpc_class, 91.5% at subclass and 98.9% at main group,
    with a median relative difference of exactly zero — so a build that divided by
    it would look right in the typical field and wrong only in the tail (worst
    0.66% / 5.88% / 50.0%, measured 2026-08-05). It is caught because the check
    takes a max and not a mean.

    `position` is a plain sort order, always unique, used only to find the
    second-highest weight for the margin. It is not a rank and is not stored.
    `dense_position` is the same ordering with ties collapsed — the k-th distinct
    weight has dense_position k — used only to find the second DISTINCT weight
    for margin_dense. It is always dense whatever dominance.rank_tie_rule says,
    because that key decides how the RANK column reads, not which runner-up the
    dense margin divides by. Not stored either.
    """
    return f"""
        SELECT
            inventor_id,
            field_id,
            year,
            window_first_year,
            inventor_weight_in_window,
            inventor_patents_in_window,
            sum(inventor_weight_in_window) OVER (PARTITION BY field_id, year)
                AS field_weight_in_window,
            inventor_weight_in_window
                / sum(inventor_weight_in_window) OVER (PARTITION BY field_id, year)
                AS share,
            {RANK_EXPRESSIONS[cfg.dominance.rank_tie_rule]} AS rank,
            count(*) OVER (PARTITION BY field_id, year, inventor_weight_in_window)
                AS n_tied_at_rank,
            count(*) OVER (PARTITION BY field_id, year) AS inventors_in_field_window,
            row_number() OVER (PARTITION BY field_id, year
                               ORDER BY inventor_weight_in_window DESC, inventor_id)
                AS position,
            dense_rank() OVER (PARTITION BY field_id, year
                               ORDER BY inventor_weight_in_window DESC)
                AS dense_position
        FROM windows_t
    """


def dominance_margin(cfg):
    """
    The dominance margin: the #1 inventor's share divided by the #2 inventor's.

    THIS IS THE DEFINITION, and it is the one place in Task 2 where the
    economics decides the code rather than the other way round. Change it here.

    Why the margin exists (task list, Task 2): a share of 4% means something
    very different when the next person holds 3.8% than when they hold 1%. The
    margin captures "towering figure", and because it is a ratio of two shares
    in the same field it barely depends on how wide the field is drawn — which
    is what makes it useful when the field width is exactly what is in doubt.

    Three decisions are baked in here.

    First, the margin is a property of the FIELD-YEAR, not of a person. It is
    computed once per field-year and attached to whoever holds rank 1. The
    alternative — dividing each inventor's weight by the next row's — would give
    the last member of a tied group a margin above 1 purely because of where the
    sort put them.

    Second, a TIED top has a margin of exactly 1.0. `position` 1 and 2 both fall
    inside the tied group, so top_weight = second_weight and the ratio is 1. This
    is deliberate and it is the economically honest answer: if two people are
    exactly level, neither towers over anything. It also means Task 3's
    alternative rule (rank 1, margin >= 2, share >= 2%) excludes tied tops on its
    own, with no special case. Ties are not rare — 17% of field-years at CPC
    subclass and 40% at main group — so this rule does real work.

    Third, a field-year with only ONE inventor has no #2 at all. The margin is
    then undefined, and dominance.margin_when_no_runner_up decides what to store:
    `missing` writes nothing and sets has_runner_up = false; `cap` writes the
    margin cap, which would place the flimsiest field-years in the whole project
    at the very top of the margin distribution. `missing` is the default for
    that reason.

    A computed value is stored UNCAPPED — rule 4, store continuous and filter
    late, not rule 2, which is the as_of discipline. dominance.margin_cap is read
    on the next line, but only under margin_when_no_runner_up: `cap`, and only to
    fill a margin that does not exist. Nothing clips a margin that does.

    THE SECOND DEFINITION (advisor, 2026-09-15). `margin_dense` divides the top
    weight by the second DISTINCT weight — `dense_position` 2 — so tied
    co-leaders count as one leader and can clear a margin threshold together:
    two people at 3 over a runner-up at 1 read 1.0 under the first definition
    and 3.0 under the second. Where only one distinct weight exists (a lone
    inventor, or everyone level) there is no dense runner-up and the same
    `margin_when_no_runner_up` rule applies. The first column is not modified by
    a bit; both are stored, and which one a rule tests is the rule's business.
    """
    if cfg.dominance.margin_when_no_runner_up == "cap":
        margin = f"COALESCE(top_weight / second_weight, {cfg.dominance.margin_cap})"
        dense = f"COALESCE(top_weight / second_dense_weight, {cfg.dominance.margin_cap})"
    else:
        margin = "top_weight / second_weight"
        dense = "top_weight / second_dense_weight"

    return f"""
        SELECT
            field_id,
            year,
            {margin}                          AS margin_uncapped,
            second_weight IS NOT NULL         AS has_runner_up,
            {dense}                           AS margin_dense,
            second_dense_weight IS NOT NULL   AS has_dense_runner_up
        FROM (
            SELECT
                field_id,
                year,
                max(inventor_weight_in_window) FILTER (WHERE position = 1)       AS top_weight,
                max(inventor_weight_in_window) FILTER (WHERE position = 2)       AS second_weight,
                max(inventor_weight_in_window) FILTER (WHERE dense_position = 2) AS second_dense_weight
            FROM ranked
            GROUP BY 1, 2
        )
    """


def write_inventor_field_years(con, cfg, width):
    """The main table. Written straight out rather than materialised first, to stay in memory."""
    con.sql(add_share_rank_and_position(cfg)).to_view("ranked", replace=True)
    con.sql(dominance_margin(cfg)).to_view("margins", replace=True)

    # The margin is attached only to rank-1 rows. A seventh-placed inventor does
    # not have a dominance margin, and storing one for them would invite it into
    # a later average by accident.
    table = con.sql("""
        SELECT
            r.inventor_id,
            r.field_id,
            r.year,
            r.window_first_year,
            r.inventor_weight_in_window,
            r.inventor_patents_in_window,
            r.field_weight_in_window,
            r.share,
            r.rank,
            r.n_tied_at_rank,
            r.rank = 1                                          AS is_top,
            CASE WHEN r.rank = 1 THEN m.margin_uncapped END     AS margin_uncapped,
            CASE WHEN r.rank = 1 THEN m.has_runner_up  END      AS has_runner_up,
            CASE WHEN r.rank = 1 THEN m.margin_dense END        AS margin_dense,
            CASE WHEN r.rank = 1 THEN m.has_dense_runner_up END AS has_dense_runner_up
        FROM ranked r
        JOIN margins m ON m.field_id = r.field_id AND m.year = r.year
    """)
    return io.write_parquet(
        con, table, inventor_file(cfg, width), cfg,
        inputs=[interim(cfg, "01_inventor_patents.parquet").as_posix()],
        step="03_measure_dominance",
    )


def write_field_years(con, cfg, width, inventor_path):
    """
    One row per field-year, aggregated FROM the file just written.

    Built by reading the inventor table back rather than recomputing, so the two
    files cannot disagree about a field's size or its top share. The patent head
    count is the exception: it is a separate quantity with a separate purpose.
    """
    io.read_parquet(con, inventor_path).to_view("inventor_field_years", replace=True)
    build_field_patent_counts(con, cfg, width)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE field_years_unflagged AS
        SELECT
            i.field_id,
            i.year,
            any_value(i.window_first_year)         AS window_first_year,
            any_value(i.field_weight_in_window)    AS field_weight_in_window,
            any_value(p.field_patents_in_window)   AS field_patents_in_window,
            count(*)                               AS inventors_in_field_window,
            max(i.share)                           AS top_share,
            max(i.n_tied_at_rank) FILTER (WHERE i.rank = 1) AS n_tied_at_top,
            max(i.margin_uncapped)                 AS margin_uncapped,
            bool_or(i.has_runner_up)               AS has_runner_up,
            max(i.margin_dense)                    AS margin_dense,
            bool_or(i.has_dense_runner_up)         AS has_dense_runner_up
        FROM inventor_field_years i
        LEFT JOIN field_patent_counts p ON p.field_id = i.field_id AND p.year = i.year
        GROUP BY 1, 2
    """)
    # The floor is a property of the DISTRIBUTION of field-year sizes, so it is
    # computed over the table just built and then attached to every row of it.
    con.sql(field_floor_sql(cfg, "field_years_unflagged")).to_view("field_floor", replace=True)
    table = con.sql(f"""
        SELECT
            u.*,
            f.floor_patents,
            u.field_patents_in_window {floor_operator(cfg)} f.floor_patents AS clears_floor
        FROM field_years_unflagged u
        JOIN field_floor f ON f.year = u.year
    """)
    return io.write_parquet(con, table, field_file(cfg, width), cfg,
                            inputs=[inventor_path.as_posix()], step="03_measure_dominance")


def floor_operator(cfg):
    """`>=` or `>`: whether a field-year holding EXACTLY the floor is kept (C-27)."""
    return ">=" if cfg.dominance.min_field_patents_inclusive else ">"


def field_floor_sql(cfg, field_years_table):
    """
    One row per window-year: the number of window patents a field-year needs.

    THIS IS THE ONE PLACE THE FLOOR IS COMPUTED. Every later step reads the
    `floor_patents` and `clears_floor` columns this attaches and computes nothing,
    which is what makes "both families use the same floored field-years" true by
    construction rather than by a check (advisor, 2026-09-15).

      absolute    dominance.min_field_patents, the same number in every year —
                  the primary specification, "fewer than ~100 patents" [TL 3]
      percentile  the dominance.floor_percentile-th percentile of field-year
                  window sizes at this width, linearly interpolated
                  (quantile_cont), so 100 is replaced by "the size a field-year
                  needs to be larger than p% of field-years at this width":
                    pooled    one number, over every field-year whose year lies
                              in deaths.year_min..year_max — the measurement
                              years, so that the floor is a fact about the
                              population the deaths are drawn from — applied to
                              every window-year including those outside it
                    per_year  the percentile within each window-year, so a
                              growing field's floor rises with it

    Why per year is offered at all: fields grow over time, so a pooled quantile
    partly tracks calendar time (02b's field-size figures say the same). Why the
    percentile is stored as a number of patents rather than only as a flag:
    every downstream table can then print what the floor WAS, and a 25th
    percentile that turns out to be 7 patents at cpc_main_group is visible
    rather than hidden behind "p25".

    The population is the field-year table itself — field-years with at least one
    inventor in the spine — because that is the table the flag is stored on. The
    two field-years the citation step (outside this extract) found with a patent
    but no inventor (C-58 D115) are
    outside it, and outside every floor.
    """
    dominance = cfg.dominance
    if dominance.floor_mode == "absolute":
        return f"""
            SELECT DISTINCT year, {dominance.min_field_patents}::DOUBLE AS floor_patents
            FROM {field_years_table}
        """
    fraction = dominance.floor_percentile / 100.0
    if dominance.floor_scope == "pooled":
        return f"""
            SELECT DISTINCT y.year, q.floor_patents
            FROM {field_years_table} y,
                 (SELECT quantile_cont(field_patents_in_window, {fraction}) AS floor_patents
                  FROM {field_years_table}
                  WHERE year BETWEEN {cfg.deaths.year_min} AND {cfg.deaths.year_max}) q
        """
    return f"""
        SELECT year, quantile_cont(field_patents_in_window, {fraction}) AS floor_patents
        FROM {field_years_table}
        GROUP BY year
    """


def check_field_size_method_is_supported(cfg):
    """
    Refuse a counting.field_size_method this step cannot honour.

    The key is validated in src/lib/config.py against `fractional | full` and read
    by nothing in src/ — build_field_patent_counts() writes count(*), which is the
    `full` reading, hardcoded. So the config currently agrees with the code by luck,
    and setting the other value would change nothing while appearing to. That is
    rule 1 broken in the quiet direction: not a magic number in code, but a setting
    with no code behind it.

    Deleting the key is the right end state and it moves the config hash, which
    forces a thirty-minute rebuild of steps 01 and 02 for data unchanged by
    construction (item 6's bill). So the value is refused instead, in the idiom
    dominance.compute_citation_share already uses in main(), and the deletion is
    queued for the Checkpoint 2 freeze. Found at Task 2b item 9; see C-38 D56.

    A fractional field size is in any case not what the task list asks for: the
    tiny-field floor is "fewer than ~100 patents (full count)", a head count
    whatever the counting method for shares.
    """
    if cfg.counting.field_size_method != "full":
        raise NotImplementedError(
            f"counting.field_size_method is '{cfg.counting.field_size_method}', but the only "
            f"field size this step computes is a plain head count of patents — see "
            f"build_field_patent_counts(). Nothing reads the key, so a value other than "
            f"'full' would be ignored rather than honoured, and Task 3's tiny-field floor "
            f"would not mean what the config says. Set it to 'full', or implement the "
            f"fractional field size and delete this guard."
        )


def build_field_patent_counts(con, cfg, width):
    """
    A plain head count of the patents in each field-window.

    This is a THIRD quantity, and keeping it apart from the two weight columns is
    the whole point of naming it this way. It is what Task 3's tiny-field
    exclusion counts ("fewer than ~100 patents in the 5-year window") and what
    the supplementary Checkpoint 1 figures split on. It must never be used as the
    denominator of a share: under fractional counting with primary codes it
    happens to equal the field's weight total, so substituting it would be
    invisible today and wrong in every robustness run.

    It counts patents, not inventor-patent pairs, so the patents Task 1 found
    with no inventor record at all are included here — a patent with an unnamed
    inventor is still a patent in the field — while contributing to nobody's
    share.

    counting.field_size_method says this is a head count, and this function does
    not read it — the count(*) below IS the setting, written out. That is why
    check_field_size_method_is_supported() refuses the other value rather than
    letting it pass unhonoured. Found at Task 2b item 9; see C-38 D56.
    """
    span = cfg.windows.dominance_window_years
    con.sql(f"""
        SELECT f.field_id, p.year AS filing_year, p.patent_id
        FROM patent_fields_at_width f
        JOIN read_parquet('{interim(cfg, "01_patents.parquet").as_posix()}') p
          ON p.patent_id = f.patent_id
    """).to_view("patents_in_fields", replace=True)

    con.sql(f"""
        SELECT field_id,
               filing_year + step.years_ahead AS year,
               count(*)                       AS field_patents_in_window
        FROM patents_in_fields,
             (SELECT unnest(range(0, {span})) AS years_ahead) step
        WHERE filing_year + step.years_ahead
              BETWEEN {cfg.dominance.first_compute_year} AND {cfg.dominance.last_compute_year}
        GROUP BY 1, 2
    """).to_view("field_patent_counts", replace=True)


# ---------------------------------------------------------------------------
# 4. Validation — part of the deliverable, not an optional extra
# ---------------------------------------------------------------------------

def validate(con, cfg, width, inventor_path, field_path, spine_note):
    """
    Every check the advisor asked for, run against the files actually written.

    Returns a list of (name, passed, detail). A failure stops the run: a wrong
    dominance table would be inherited by every later result in the thesis.
    """
    io.read_parquet(con, inventor_path).to_view("check_inventor", replace=True)
    io.read_parquet(con, field_path).to_view("check_field", replace=True)
    results = []

    # 1. Shares within a field-year sum to 1. The strongest single check there
    #    is: it fails if the denominator is wrong, if patents were double-counted
    #    across CPC codes, or if the two counting methods were mixed.
    #    Note that this holds for FULL counting too, not only fractional — if the
    #    denominator is the sum of the same weights over the same rows, the
    #    shares sum to one whatever the weights mean. It is a same-counting
    #    check, not a fractional-counting check.
    worst = con.sql("""
        SELECT max(abs(total - 1.0)) FROM (
            SELECT field_id, year, sum(share) AS total FROM check_inventor GROUP BY 1, 2)
    """).fetchone()[0]
    #    `worst is not None` makes an EMPTY table fail rather than pass, which is
    #    the right answer: no field-years at all means the build produced nothing.
    #    The detail string has to guard the same way — it formatted `worst` on the
    #    line below unconditionally, so an empty table raised TypeError here and
    #    the guard beside it could never actually report. Found by Task 2b item 8,
    #    which is the first test to run this function; see notes/decisions.md C-37.
    results.append((
        "shares sum to 1 within every field-year",
        worst is not None and worst < 1e-9,
        f"worst deviation {worst:.3e} across all field-years" if worst is not None
        else "NO field-years at all — the inventor table is empty",
    ))

    # 2. No share above 1, none negative.
    bad = con.sql("SELECT count(*) FROM check_inventor WHERE share < 0 OR share > 1").fetchone()[0]
    results.append(("every share lies in [0, 1]", bad == 0, f"{bad:,} rows outside [0, 1]"))

    # 3. Rank 1 exists everywhere, and is unique wherever the top is untied.
    #    Under the `competition` and `dense` rules a tie genuinely produces
    #    several rank-1 inventors, so "exactly one" is only required where the
    #    top weight is not shared. The tie rate is reported because it is itself
    #    a finding about how thin these fields are.
    no_top, tied, total = con.sql("""
        SELECT count(*) FILTER (WHERE tops = 0),
               count(*) FILTER (WHERE tops > 1),
               count(*)
        FROM (SELECT field_id, year, count(*) FILTER (WHERE rank = 1) AS tops
              FROM check_inventor GROUP BY 1, 2)
    """).fetchone()
    #    The tie RATE needs a denominator, and an empty table has none. Guarded
    #    for the same reason as check #1 above: a report that raises instead of
    #    reporting tells the reader nothing about why the build is empty.
    tie_rate = f"({tied / total:.1%})" if total else "(no field-years)"
    results.append((
        "every field-year has a rank-1 inventor, unique where the top is untied",
        no_top == 0,
        f"{no_top:,} field-years with no rank-1; {tied:,} of {total:,} "
        f"{tie_rate} have a TIED top under rule '{cfg.dominance.rank_tie_rule}'",
    ))

    # 4. Every margin that exists is at least 1, by construction.
    below, missing, n_top = con.sql("""
        SELECT count(*) FILTER (WHERE margin_uncapped < 1.0),
               count(*) FILTER (WHERE margin_uncapped IS NULL),
               count(*)
        FROM check_inventor WHERE rank = 1
    """).fetchone()
    results.append((
        "every stored margin is >= 1",
        below == 0,
        f"{below:,} below 1; {missing:,} of {n_top:,} rank-1 rows have no margin "
        f"(field-years with a single inventor and no runner-up)",
    ))

    # 4b. The second margin definition can only be weakly larger than the first
    #    — the second distinct weight is at most the second row's — and must
    #    equal it wherever the top is untied, because then the second row IS the
    #    second distinct weight. And a dense runner-up that exists is strictly
    #    below the top, so the dense margin is strictly above 1 wherever it has
    #    one. Any of the three failing means the two columns are not the two
    #    definitions they claim to be.
    dense_bad = con.sql("""
        SELECT count(*) FROM check_inventor
        WHERE rank = 1 AND (
              margin_dense < margin_uncapped
           OR (n_tied_at_rank = 1 AND abs(margin_dense - margin_uncapped) > 1e-12)
           OR (has_dense_runner_up AND margin_dense <= 1.0))
    """).fetchone()[0]
    results.append((
        "margin_dense is never below margin_uncapped and equals it where the top is untied",
        dense_bad == 0,
        f"{dense_bad:,} rank-1 rows where the two margin definitions disagree in a way "
        f"neither definition allows",
    ))

    # 4c. The floor flag is the test written out: floor_patents present on every
    #    row, and clears_floor exactly `size {op} floor_patents`. A NULL anywhere
    #    would be read downstream as "not floored", silently.
    floor_bad = con.sql(f"""
        SELECT count(*) FROM check_field
        WHERE floor_patents IS NULL OR clears_floor IS NULL
           OR clears_floor IS DISTINCT FROM
              (field_patents_in_window {floor_operator(cfg)} floor_patents)
    """).fetchone()[0]
    results.append((
        "clears_floor agrees with floor_patents on every field-year",
        floor_bad == 0,
        f"{floor_bad:,} field-years where the stored flag is not the stored floor applied "
        f"to the stored size (floor_mode {cfg.dominance.floor_mode})",
    ))

    # 4d. The stored floor is the one THIS configuration implies, recomputed
    #    from the written table. Under absolute mode that is the configured
    #    number on every row; under percentile mode the quantile over the file.
    #    This is the check that catches a field-year table written by another
    #    floor configuration and read by this one.
    con.sql(field_floor_sql(cfg, "check_field")).to_view("check_floor", replace=True)
    floor_stale = con.sql("""
        SELECT count(*) FROM check_field c
        LEFT JOIN check_floor f ON f.year = c.year
        WHERE f.floor_patents IS NULL OR abs(c.floor_patents - f.floor_patents) > 1e-9
    """).fetchone()[0]
    floor_values = con.sql(
        "SELECT min(floor_patents), max(floor_patents) FROM check_field").fetchone()
    results.append((
        "floor_patents is the floor this configuration implies",
        floor_stale == 0,
        f"{floor_stale:,} field-years carry a floor other than this config's; stored floor "
        f"runs from {floor_values[0]} to {floor_values[1]} patents",
    ))

    # 5. Reconciliation with Task 1: the total credit handed out must not have
    #    changed. Compared against filing years, not window years, since a
    #    patent enters five windows on purpose.
    here = con.sql("SELECT round(sum(weight_in_year), 4) FROM activity").fetchone()[0]
    task1 = con.sql(f"""
        SELECT round(sum(weight), 4)
        FROM read_parquet('{interim(cfg, "01_inventor_patents.parquet").as_posix()}')
    """).fetchone()[0]
    results.append((
        "total counting weight reconciles with Task 1",
        abs(here - task1) < 1e-3,
        f"Task 2 at {width}: {here:,.4f} — Task 1 spine: {task1:,.4f}",
    ))

    # 6. No duplicate keys. A duplicate would double an inventor's share without
    #    breaking any of the checks above.
    dupes = con.sql("""
        SELECT count(*) FROM (SELECT inventor_id, field_id, year FROM check_inventor
                              GROUP BY 1, 2, 3 HAVING count(*) > 1)
    """).fetchone()[0]
    results.append(("one row per inventor x field x year", dupes == 0,
                    f"{dupes:,} duplicated keys"))

    # 7. The two files agree about every field-year.
    disagreement = con.sql("""
        SELECT count(*) FROM (
            SELECT field_id, year FROM check_inventor GROUP BY 1, 2
            EXCEPT SELECT field_id, year FROM check_field)
    """).fetchone()[0]
    results.append(("the field-year file covers every field-year", disagreement == 0,
                    f"{disagreement:,} field-years missing"))

    # 8. The no-look-ahead guard and the spine identity, both already run during
    #    the build, recorded here so the report is complete.
    results.append(("no patent enters a window that closed before it was filed", True,
                    "verified row by row by src/lib/asof.py during the build"))
    results.append(("the counting weight matches src/02_build_patent_tables.py", True, spine_note))

    return results


# ---------------------------------------------------------------------------

def summarise(con, cfg, width, field_path):
    """A few numbers per width, so the run reports a result rather than 'done'."""
    io.read_parquet(con, field_path).to_view("summary_field", replace=True)
    row = con.sql("""
        SELECT count(DISTINCT field_id), count(*),
               median(field_patents_in_window),
               quantile_cont(top_share, 0.5), quantile_cont(top_share, 0.9),
               max(top_share),
               min(floor_patents), max(floor_patents),
               count(*) FILTER (WHERE clears_floor)
        FROM summary_field
    """).fetchone()
    return {"width": width, "fields": row[0], "field_years": row[1],
            "median_field_patents": row[2], "top_share_p50": row[3],
            "top_share_p90": row[4], "top_share_max": row[5],
            "floor_min": row[6], "floor_max": row[7], "field_years_above_floor": row[8]}


def write_report(cfg, all_results, summaries, timings):
    lines = [
        "# Task 2 — dominance for every inventor: validation",
        "",
        f"Config `{cfg.name}`, hash `{cfg.hash}`. Written by `src/03_measure_dominance.py`.",
        "",
    ]
    if cfg.sample_mode:
        lines += [
            f"> **Sample mode is ON** — CPC section `{cfg.sample_cpc_section}` only. "
            f"These are debugging numbers, not results.",
            "",
        ]
    lines += [
        f"Window: {cfg.windows.dominance_window_years} years, t-"
        f"{cfg.windows.dominance_window_years - 1} through t, for window-years "
        f"{cfg.dominance.first_compute_year}–{cfg.dominance.last_compute_year}. "
        f"Counting: **{cfg.counting.method}**. CPC assignment: **{cfg.cpc.assignment}**. "
        f"Tie rule: **{cfg.dominance.rank_tie_rule}**. Margins stored **uncapped**.",
        "",
        "## What was built",
        "",
        "| width | fields | field-years | median field size | top share p50 | p90 | max | "
        "floor (patents) | field-years above it | runtime |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        floor = (f"{s['floor_min']:g}" if s["floor_min"] == s["floor_max"]
                 else f"{s['floor_min']:g}–{s['floor_max']:g} by year")
        lines.append(
            f"| `{s['width']}` | {s['fields']:,} | {s['field_years']:,} | "
            f"{s['median_field_patents']:,.0f} | {s['top_share_p50']:.4f} | "
            f"{s['top_share_p90']:.4f} | {s['top_share_max']:.4f} | {floor} | "
            f"{s['field_years_above_floor']:,} | {timings[s['width']]:.0f}s |"
        )
    spec = (f"absolute, `dominance.min_field_patents` = {cfg.dominance.min_field_patents}"
            if cfg.dominance.floor_mode == "absolute"
            else f"the {cfg.dominance.floor_percentile:g}th percentile of field-year window "
                 f"sizes, {cfg.dominance.floor_scope}")
    lines += [
        "",
        f"The field floor is **{spec}**, {'at or above' if cfg.dominance.min_field_patents_inclusive else 'above'} "
        f"which a field-year clears it. It is stored on every field-year as `floor_patents` "
        f"and `clears_floor` and applied by no step before Task 3; the column above says what "
        f"it came to in patents.",
    ]

    lines += ["", "## Validation", ""]
    for width, results in all_results.items():
        lines += [f"### `{width}`", "", "| check | result | detail |", "| --- | :---: | --- |"]
        for name, passed, detail in results:
            lines.append(f"| {name} | {'PASS' if passed else '**FAIL**'} | {detail} |")
        lines.append("")
    return io.write_output_text(cfg, "task2_validation.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild, ignoring cached outputs")
    args = parser.parse_args()

    cfg = config_module.load(args.config)
    io.ensure_dirs(cfg)
    # Before anything is rewritten: say so if a report was edited by hand
    # since the last run. The stamped copy is authoritative, so this is a
    # warning and not an error (src/lib/io.py, and notes/decisions.md C-24).
    io.warn_about_hand_edited_outputs(cfg)
    Path(cfg.runtime.duckdb_temp_directory).mkdir(parents=True, exist_ok=True)

    funnel = Funnel("03_measure_dominance", cfg)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{cfg.runtime.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory = '{cfg.runtime.duckdb_temp_directory}'")
    if cfg.runtime.duckdb_threads:
        con.execute(f"SET threads = {cfg.runtime.duckdb_threads}")

    print(f"Task 2 — config '{cfg.name}' (hash {cfg.hash}), "
          f"sample_mode={'ON: CPC section ' + cfg.sample_cpc_section if cfg.sample_mode else 'off'}")
    print(f"Widths: {', '.join(cfg.task2.widths)}. Memory limit "
          f"{cfg.runtime.duckdb_memory_limit}, spilling to {cfg.runtime.duckdb_temp_directory}.")
    # The filenames in data/interim/ do not carry the config hash, so nothing
    # about 01_patents.parquet says whether it holds one CPC section or all of
    # them. Reading Task 1's sample-mode output during a full run would produce
    # a complete, plausible set of results for a thirteenth of the data.
    io.require_built_by_this_config(
        [interim(cfg, "01_inventor_patents.parquet"), interim(cfg, "01_patents.parquet")],
        cfg, rebuild_with="python src/02_build_patent_tables.py --force",
    )

    check_field_size_method_is_supported(cfg)

    if cfg.dominance.compute_citation_share:
        raise NotImplementedError(
            "dominance.compute_citation_share is true, but citation-share dominance is a Task 6 "
            "measure and is not built yet. Set it to false."
        )

    all_results, summaries, timings = {}, [], {}
    produced = []

    for width in cfg.task2.widths:
        started = time.perf_counter()
        print(f"\n--- {width} ---")

        spine = build_spine_at_width(con, cfg, width, funnel)
        spine.to_view("spine_source", replace=True)
        con.execute("CREATE OR REPLACE TEMP TABLE spine_at_width AS SELECT * FROM spine_source")
        spine_note = check_matches_task_one_spine(con, cfg, width)
        print(f"  spine: {spine_note}")

        build_activity(con, cfg, funnel, width)
        windows = build_windows(con, cfg, funnel, width)
        windows.to_view("windows_source", replace=True)
        con.execute("CREATE OR REPLACE TEMP TABLE windows_t AS SELECT * FROM windows_source")

        inventor_path = write_inventor_field_years(con, cfg, width)
        field_path = write_field_years(con, cfg, width, inventor_path)
        produced += [inventor_path, field_path]

        timings[width] = time.perf_counter() - started
        all_results[width] = validate(con, cfg, width, inventor_path, field_path, spine_note)
        summaries.append(summarise(con, cfg, width, field_path))

        rows = con.sql(f"SELECT count(*) FROM read_parquet('{inventor_path.as_posix()}')").fetchone()[0]
        size = inventor_path.stat().st_size / 1e9
        print(f"  {rows:,} inventor-field-year rows, {size:.2f} GB, {timings[width]:.0f}s")
        for name, passed, detail in all_results[width]:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")

    stamped, plain = write_report(cfg, all_results, summaries, timings)
    io.write_latest(cfg, "03_measure_dominance", [stamped] + [p.name for p in produced],
                    summary="Task 2: dominance for every inventor at three field widths.")
    funnel.finish()

    failures = [(w, n) for w, rs in all_results.items() for n, ok, _ in rs if not ok]
    if failures:
        raise SystemExit(
            f"\n{len(failures)} validation check(s) FAILED: "
            + "; ".join(f"{w}/{n}" for w, n in failures)
            + "\nThe dominance tables are on disk but must not be used."
        )
    print(f"\nAll checks passed. Report: {plain}")


if __name__ == "__main__":
    main()
