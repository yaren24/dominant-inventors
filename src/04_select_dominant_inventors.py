"""
Step 04 — Task 3: count the dominant inventors.

Reads : data/interim/02_inventor_field_years_<width>.parquet   (share, rank, margin)
        data/interim/02_field_years_<width>.parquet            (the floor's column)
        data/interim/01_inventor_patents.parquet               (filing years, §3.3 only)
Writes: data/interim/03_dominance_candidates_<width>.parquet
        outputs/<config_name>/task3_dominant_inventors.md
        outputs/<config_name>/task3_cells.csv
        outputs/<config_name>/task3_field_size.csv
        outputs/<config_name>/task3_pairs_<width>.csv
        outputs/<config_name>/task3_persistence.png
        logs/runs/<timestamp>_04_select_dominant_inventors/

Config keys used: task2.widths, dominance.cutoffs, dominance.cutoff,
dominance.alternative_rule.{max_rank, min_margin, min_share} (applied twice, to
margin_uncapped and to margin_dense — the second is the advisor's tie-blind
definition of 2026-09-15),
dominance.floor_mode, dominance.floor_percentile, dominance.floor_scope,
dominance.min_field_patents_inclusive (all four only to DESCRIBE the floor; the
flag itself is read from step 03's field-year table and never recomputed),
dominance.apply_min_field_size, dominance.report_both_min_field_size,
dominance.metric, dominance.first_compute_year, dominance.rank_tie_rule,
dominance.margin_when_no_runner_up, deaths.year_min, deaths.year_max,
windows.dominance_window_years, counting.method, cpc.assignment, fields.strategy,
checkpoints.share_decimals, sample_mode, sample_cpc_section, runtime.*

Task in the task list: Task 3. This is the FIRST step that applies a dominance
cutoff. Task 2 stored everything continuous on purpose; here the cutoffs from
`dominance.cutoffs` are swept so that Checkpoint 2 can choose one.

Serves: CHECKPOINT 2, and through it SQ1 and SQ5. It does NOT choose a field
width or a cutoff — that is a joint decision with the advisor, and
`dominance.cutoff`'s own comment in the config says 0.05 has no authority until
then.

What the counting rule means, exactly
-------------------------------------
A pair is one inventor and one field. It is COUNTED at cutoff c if there is at
least one window-year t in the death window where the inventor's share of that
field reached c. With the tiny-field exclusion the cutoff and the floor must be
cleared in the SAME window — clearing the cutoff only in years when the field was
thin does not count, even if the field is large in other years. That is
structural here rather than promised: `year` is the window's END year in both
Task 2 tables, so joining on (field_id, year) joins the same five-year window and
there is no year range that could be written the wrong way round.

Three things this step reports that the task list did not ask for, because
measuring the thing it did ask for showed the question was underspecified
--------------------------------------------------------------------------
1. EXPOSURE — the total number of qualifying (pair, window-year) observations.
   The task list asks for the median number of years a counted pair spends above
   the cutoff, "because Task 4 will require dominance at death, so persistence
   tells us in advance how much that requirement will cost." The median cannot
   do that job and the total can: under Task 4's strict rule a pair is treated by
   a death in year t exactly when it qualifies in window-year t, so summing
   qualifying pair-years counts treatment opportunities. The median is reported
   as asked, beside it.
2. THE LOOSE MULTIPLIER — the same total under Task 4a's loose rule (qualifying
   in any of the five windows ending in the death year or the four before it),
   divided by the strict one. Task 4a's two-timing-rules question, answered here
   with no death data at all.
3. THE CAREER SHARE — the median share over every window-year the pair appears in
   the field. Persistence measured in overlapping windows cannot mean durability:
   consecutive windows share four filing years, so ONE heavy filing year occupies
   up to five window-years. Measured, the median is 3 — below that footprint —
   and it is flat at 3.0-3.4 across every cutoff and width, so it cannot separate
   the cutoffs it was asked to separate. §3.3 shows what the pairs actually are.

What this step does NOT do: it does not touch deaths. The forward count, the age
columns and the 4c identity flags are Task 4. It does not re-derive the full-count
arm either — Task 2b item 9 already emitted these cells under both counting
methods, and tests/test_task3.py pins this step's numbers to that table.

See notes/decisions.md C-39.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")          # write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from src.lib import config as config_module
from src.lib import io
from src.lib.funnel import Funnel
# floor_clause owns the inclusive/exclusive boundary of dominance.min_field_patents
# (C-27, which moves 4 / 37 / 420 field-years). Items 1, 2 and 9 all read the floor
# through that one function. Everywhere else in this repository two lines of
# duplication beat an indirection to trace; here the indirection IS the point,
# because a private copy of the boundary could drift from the one item 9's
# cross-reference table uses and the two notes would disagree about the same cell
# while both looked right. Moving it into src/lib/ is the tidier end state, changes
# no number, and belongs to the Checkpoint 2 freeze.
from src.lib.shared import (compress_years, field_years_path,
                                     floor_clause, floor_words, inventor_field_years_path,
                                     stored_floor)

THIS_STEP = "src/04_select_dominant_inventors.py"

# Ordered cutoffs need colours that read as ordered and survive being printed in
# black and white. The same light-to-dark ramp as QUARTILE_COLOURS in
# src/steps/02b_checkpoint1_figures.py, so the Task 3 figure and the Checkpoint 1
# figures read as one set. Presentation rather than an analytical choice, which is
# why it is a constant here and not a config key.
CUTOFF_COLOURS = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
REFERENCE_COLOUR = "#d62728"
ALTERNATIVE_COLOUR = "#777777"


# ---------------------------------------------------------------------------
# 1. Paths, derived years, and the guards. All of these raise; none report.
# ---------------------------------------------------------------------------

def candidates_path(cfg, width):
    return io.interim_path(cfg, f"03_dominance_candidates_{width}.parquet")


def artifact_first_year(cfg):
    """
    The first window-year the STORED artifact keeps, which is earlier than the
    first window-year Task 3 counts.

    Task 3's cells cover the death window. Task 4a's loose timing rule asks
    whether dominance held in any of the five windows ending in the death year or
    the four years before it, so a death in the first usable year reaches four
    window-years further back. Cutting the artifact at deaths.year_min would
    silently understate the loose rule for the earliest deaths — and understate it
    in exactly the direction that makes the design look less viable at the
    checkpoint where viability is decided.

    Clipped at dominance.first_compute_year because step 03 computed nothing
    before it. If the clip binds, the note says so, because Task 4's loose rule is
    then genuinely unavailable for the earliest deaths rather than merely awkward.
    """
    reach_back = cfg.deaths.year_min - (cfg.windows.dominance_window_years - 1)
    return max(reach_back, cfg.dominance.first_compute_year)


def loose_reach_is_clipped(cfg):
    """True when step 03's first computed year cuts into Task 4a's loose reach."""
    return cfg.deaths.year_min - (cfg.windows.dominance_window_years - 1) \
        < cfg.dominance.first_compute_year


def sorted_cutoffs(cfg):
    """
    dominance.cutoffs, ascending.

    Sorted explicitly rather than trusted: the config is a list a human edits, so
    it could arrive in any order, and the table columns and the monotonicity check
    would then disagree about which cutoff is "the next one up" — the check would
    fail on correct numbers.
    """
    return sorted(cfg.dominance.cutoffs)


def smallest_selectable_share(cfg):
    """
    The smallest share any Task 3 or Task 4 rule can select on.

    A row below it clears no cutoff and fails the alternative rule's share floor,
    so it cannot appear in any cell of either task's tables. That is what makes
    dropping it a filter on the stored SAMPLE rather than a discretisation of the
    measurement (rule 4), and check_pool_threshold_is_not_binding() asserts it
    instead of arguing it.
    """
    return min(list(cfg.dominance.cutoffs) + [cfg.dominance.alternative_rule.min_share])


def check_metric_is_patent_share(cfg):
    """
    Refuse a dominance.metric this step cannot honour.

    The stored `share` is a patent share whatever the key says —
    dominance.compute_citation_share is false and citation share is a Task 6
    measure two checkpoints away. A note whose table header called it a citation
    share would be a lie about what was measured.

    Same idiom as the guard in src/03_measure_dominance.py that refuses a field-size
    setting it cannot honour (C-38 D56). Named by description rather than by
    function, because tests/test_task2b_item9.py enumerates every file mentioning
    that setting's key, and a cross-reference here would widen a guard whose whole
    value is that a new file appearing in the list is a finding.
    """
    if cfg.dominance.metric != "patent_share":
        raise NotImplementedError(
            f"dominance.metric is '{cfg.dominance.metric}', but the only share stored by "
            f"src/03_measure_dominance.py is a patent share, and citation share is a Task 6 "
            f"measure that is not built. Every cutoff below would be applied to patent shares "
            f"while the note claimed otherwise. Set it to 'patent_share'."
        )


def check_alternative_rule_is_expressible(cfg):
    """
    Refuse alternative_rule.max_rank above 1.

    src/03_measure_dominance.py stores the margin as `CASE WHEN rank = 1 THEN ...`,
    so every rank-2 row carries NULL and `margin_uncapped >= 2` is NULL there,
    hence false. A max_rank of 2 would therefore select exactly ZERO rank-2 rows
    while appearing to permit them — the alternative-rule column would silently
    stay a rank-1 column. That is C-38 D56's pattern once more: a setting with no
    code behind it.

    Widening the rule means storing margins for lower ranks in Task 2, which is a
    rebuild and a Checkpoint 2 decision, not something this step can fake.
    """
    if cfg.dominance.alternative_rule.max_rank != 1:
        raise NotImplementedError(
            f"dominance.alternative_rule.max_rank is "
            f"{cfg.dominance.alternative_rule.max_rank}, but src/03_measure_dominance.py stores "
            f"margin_uncapped and margin_dense only on rank-1 rows, so every row with rank > 1 "
            f"has a NULL margin and would fail the margin test silently. The alternative-rule column would look "
            f"widened and be unchanged. Set it to 1, or store margins for lower ranks in Task 2 "
            f"first."
        )


def check_colours_cover_the_cutoffs(cfg):
    """
    Refuse more cutoffs than the figure has colours for.

    zip() would otherwise drop the last line from the figure with no error, and a
    figure quietly missing a cutoff is worse than no figure.
    """
    if len(sorted_cutoffs(cfg)) > len(CUTOFF_COLOURS):
        raise ValueError(
            f"dominance.cutoffs has {len(sorted_cutoffs(cfg))} values and CUTOFF_COLOURS has "
            f"{len(CUTOFF_COLOURS)}, so the persistence figure would silently omit the last "
            f"cutoff. Add colours to CUTOFF_COLOURS in {THIS_STEP}."
        )


def check_filing_years_are_reachable(cfg):
    """
    Refuse to derive filing-year activity when the field ids do not nest.

    §3.3 checks whether a counted pair's run of qualifying window-years sits on one
    burst of filing or on sustained activity. The filing years live in
    01_inventor_patents.parquet at cfg.fields.strategy only, and cpc_class is that
    field id's three-character prefix — but ONLY under cpc.assignment: primary,
    where a patent has one code. Under `all`, a patent carrying two subclasses of
    the same class would be counted twice in the class, inflating the very activity
    the check is about.
    """
    if cfg.cpc.assignment != "primary":
        raise NotImplementedError(
            f"cpc.assignment is '{cfg.cpc.assignment}', so a patent can carry several codes and "
            f"truncating a subclass id to its class would count one patent twice in the class. "
            f"The filing-year check in §3.3 cannot be derived from "
            f"01_inventor_patents.parquet this way. Implement it at each width, or drop the "
            f"section."
        )


def reporting_switches(cfg):
    """
    Which of the two tables this run prints, and the sentence that says why.

    Both keys exist and neither was read in the first draft of this step, which is
    rule 1 broken in the quiet direction — a setting that looks like it governs
    something and does not (C-33, C-38 D56). The task list asks for the table BOTH
    with and without the exclusion, so `report_both_min_field_size: false` is a
    departure from it and prints loudly rather than quietly.

    Returns (floor_settings_to_print, sentence).
    """
    if cfg.dominance.report_both_min_field_size:
        return [True, False], (
            f"Both tables, as the task list asks: with the tiny-field floor and without it "
            f"(`dominance.report_both_min_field_size: true`)."
        )

    only = cfg.dominance.apply_min_field_size
    print(f"\n  NOTE — dominance.report_both_min_field_size is false, so this run prints ONE "
          f"table ({'with' if only else 'without'} the tiny-field floor). The task list asks "
          f"for both. The decomposition below still reports the without-floor counts.\n")
    return [only], (
        f"**One table only.** `dominance.report_both_min_field_size` is false and "
        f"`dominance.apply_min_field_size` is {str(only).lower()}, so this run shows the "
        f"{'with' if only else 'without'}-floor table. The task list asks for both; the "
        f"decomposition in §2 still carries the without-floor counts."
    )


# ---------------------------------------------------------------------------
# 2. The rules — the one place a column of the Task 3 table is defined
# ---------------------------------------------------------------------------

def alternative_rule_sql(cfg, margin_column="margin_uncapped"):
    """
    The "towering figure" rule, as SQL. ONE place, because the cells, the
    exclusion report and the figure all need exactly these three conditions.

    `margin_column` is `margin_uncapped` (the second ROW's share: a tied top
    reads 1.0 and fails) or `margin_dense` (the second DISTINCT share: tied
    co-leaders count as one leader and can pass together). Same thresholds, same
    share floor; only the denominator's definition differs. Advisor, 2026-09-15.

    Why it exists (task list, Task 3): a share of 4% means something different
    when the runner-up holds 3.8% than when they hold 1%. The margin captures
    towering, and because it is a ratio of two shares inside one field it barely
    depends on how wide the field is drawn — useful when the width is exactly
    what is in doubt. The share floor is not decoration: in a large diffuse field
    a #1 at 0.6% against a #2 at 0.25% clears the margin while dominating
    nothing, and the tiny-field exclusion does not catch it because that field is
    not small, just spread thin.

    TWO EXCLUSIONS ARE LOAD-BEARING AND ARE REPORTED RATHER THAN RELIED ON, by
    alternative_rule_exclusions() below:

      - a TIED top has margin exactly 1.0 under dominance.rank_tie_rule:
        competition (see dominance_margin() in src/03_measure_dominance.py), so
        ties fail `margin >= 2` with no special case. Right — if two people are
        exactly level, neither towers — and it does real work: ties are 39.5% of
        main-group field-years.
      - a rank-1 inventor who is the ONLY inventor in the field-window has NULL
        margin under dominance.margin_when_no_runner_up: missing, and
        `NULL >= 2.0` is NULL, hence false. So a person holding 100% of a field
        FAILS this rule. Also right — nobody towers over an empty field — but it
        is contingent on a Task 2 storage key rather than on the rule, and under
        `cap` those same monopolists return with the margin cap and pass. Item 1
        showed the floor removes all 6 / 131 / 10,348 such field-years, so in the
        with-floor table the exclusion is empty.

    dominance.margin_cap is NOT read here. The stored margin is uncapped (rule 4)
    and the cap is presentation only, in step 02b, the Checkpoint 1 figures step,
    which is outside this extract: a #1 at forty times the
    runner-up clears min_margin on the uncapped value, which is the right reading.
    """
    rule = cfg.dominance.alternative_rule
    return (f"rank <= {rule.max_rank} "
            f"AND {margin_column} >= {rule.min_margin} "
            f"AND share >= {rule.min_share}")


def rules(cfg):
    """
    One entry per column of the Task 3 table: the share cutoffs, then the
    alternative rule.

    `id` becomes a SQL column name in the per-year table and a key in the CSV, so
    it has to be a legal identifier and stable across runs — hence the integer
    basis-point form rather than formatting a float.

    The cutoff comparison is `>=`, inclusive: a share of exactly 5% is dominant
    under a 5% rule. Unlike the floor's boundary that is not a config key, and
    C-30 established that exact ties do occur on the fractional lattice even in
    large fields, so it is not automatically a null choice. Measured on the real
    data it moves 1 qualifying observation at 5% and 4 at 2.5%, both at main
    group, so it decides almost nothing — but it is a choice, it is recorded in
    C-39 D64 and pinned by a test, and the key is queued for the Checkpoint 2
    freeze rather than added now at the cost of a rebuild of steps 01 and 02.
    """
    entries = [
        {"id": f"cutoff_{int(round(cutoff * 10000)):04d}",
         "label": f"share >= {cutoff:.2%}".replace(".00%", "%"),
         "kind": "cutoff",
         "cutoff": cutoff,
         "test": f"share >= {cutoff}"}
        for cutoff in sorted_cutoffs(cfg)
    ]
    rule = cfg.dominance.alternative_rule
    entries.append({
        "id": "alternative",
        "label": f"rank {rule.max_rank}, margin >= {rule.min_margin:g}, "
                 f"share >= {rule.min_share:.1%}",
        "kind": "alternative",
        "cutoff": None,
        "test": alternative_rule_sql(cfg),
    })
    # The same rule on the tie-blind margin. A separate column rather than a
    # setting that swaps the definition, so the two are always reported side by
    # side and neither can quietly become the other (advisor, 2026-09-15).
    entries.append({
        "id": "alternative_dense",
        "label": f"rank {rule.max_rank}, margin_dense >= {rule.min_margin:g}, "
                 f"share >= {rule.min_share:.1%}",
        "kind": "alternative",
        "cutoff": None,
        "test": alternative_rule_sql(cfg, "margin_dense"),
    })
    return entries


# ---------------------------------------------------------------------------
# 3. The candidate pool: the only pass over a big table
# ---------------------------------------------------------------------------

def candidate_rows_sql(cfg, width):
    """
    THE SAME-WINDOW JOIN, and the one place it is written.

    The join key is (field_id, year) and `year` is the window's END year in BOTH
    Task 2 tables, so joining on it joins the same five-year window by
    construction. There is no year range and therefore no inequality that could be
    written the wrong way round — window_rows_sql()'s argument in
    src/03_measure_dominance.py, one table over.

    That is a claim about two files rather than about one query, so it is CHECKED
    and not trusted: window_first_year is stored on both sides and must agree on
    every joined row (check_same_window_join). It is deliberately NOT part of the
    join condition, because an extra equality would DROP the disagreeing rows
    silently instead of reporting them — the failure mode this whole step exists to
    avoid.

    LEFT JOIN rather than inner, for the same reason. A candidate row with no
    field-year row, and a field-year row whose patent head count is NULL, are two
    different defects and an inner join hides both. write_field_years() in
    src/03_measure_dominance.py builds field_patents_in_window through a LEFT JOIN of
    its own, so NULL is reachable — and a NULL floor makes `>= 100` NULL, hence
    false, which would understate every with-exclusion cell while looking entirely
    plausible.
    """
    return f"""
        SELECT
            i.inventor_id,
            i.field_id,
            i.year,
            i.window_first_year,
            i.share,
            i.rank,
            i.n_tied_at_rank,
            i.margin_uncapped,
            i.has_runner_up,
            i.margin_dense,
            i.has_dense_runner_up,
            i.inventor_weight_in_window,
            i.inventor_patents_in_window,
            i.field_weight_in_window,
            f.field_patents_in_window,
            f.inventors_in_field_window,
            f.window_first_year AS field_window_first_year,
            -- The floor, as step 03 stored it. Read, never recomputed: under a
            -- percentile floor there is no number in the config to recompute from.
            f.floor_patents,
            f.clears_floor
        FROM candidate_source i
        LEFT JOIN read_parquet('{field_years_path(cfg, width).as_posix()}') f
               ON f.field_id = i.field_id
              AND f.year     = i.year
    """


def build_candidates(con, cfg, width, funnel):
    """
    Cut the stored panel down to the rows any Task 3 or Task 4 rule could select,
    and attach each row's own field-year.

    Two funnel filters, because two things are dropped for two different reasons
    and the diary has to be able to say which. Everything after this reads
    task3_candidates.
    """
    first, last = artifact_first_year(cfg), cfg.deaths.year_max
    smallest = smallest_selectable_share(cfg)
    span = cfg.windows.dominance_window_years

    rows = io.read_parquet(con, inventor_field_years_path(cfg, width))
    rows = funnel.filter(
        rows, f"{width}_task3_window",
        f"year BETWEEN {first} AND {last}",
        why=(
            f"Task 3 counts an inventor x field pair if it clears a cutoff at some point in "
            f"{cfg.deaths.year_min}-{cfg.deaths.year_max}, the years in which a death is usable, "
            f"so window-years outside that span cannot enter a cell. The lower bound stored here "
            f"is {first} rather than {cfg.deaths.year_min} because Task 4a's loose timing rule "
            f"asks whether dominance held in any of the {span} windows ending in the death year "
            f"or the {span - 1} years before it, so a death in {cfg.deaths.year_min} reaches back "
            f"to {first}. Task 3's own counting is confined to {cfg.deaths.year_min}-"
            f"{cfg.deaths.year_max} where the cells are formed, so those four extra window-years "
            f"change no number in this note and exist only so that Task 4 need not re-read the "
            f"47-million-row build."
        ),
    )
    rows = funnel.filter(
        rows, f"{width}_selectable_share",
        f"share >= {smallest}",
        why=(
            f"{smallest:.1%} is the smallest share any Task 3 or Task 4 rule can select on: the "
            f"lowest of dominance.cutoffs and the alternative rule's own share floor of "
            f"{cfg.dominance.alternative_rule.min_share:.1%}. A row below it clears no cutoff and "
            f"fails the alternative rule, so it cannot appear in any cell of either task's "
            f"tables, and dropping it changes no count — which is asserted rather than argued, by "
            f"check_pool_threshold_is_not_binding(). This is a filter on the SAMPLE this step "
            f"stores and not a discretisation of the measurement: every continuous column "
            f"survives on the rows kept, and the un-truncated share distribution stays in "
            f"02_inventor_field_years_{width}.parquet, which is where anything describing that "
            f"distribution must read from."
        ),
    )

    rows.to_view("candidate_source", replace=True)
    con.execute(f"CREATE OR REPLACE TEMP TABLE task3_candidates AS "
                f"{candidate_rows_sql(cfg, width)}")
    funnel.checkpoint(f"{width}: pair-years joined to their own field-year",
                      con.sql("SELECT * FROM task3_candidates"))
    return check_same_window_join(con, cfg, width)


def check_same_window_join(con, cfg, width):
    """
    Three guards on the join, all arithmetic rather than findings, so they raise.

    The first two are the LEFT JOIN's whole purpose: an unmatched row and a NULL
    head count both make the floor test false, which would understate every
    with-exclusion cell by a plausible amount. The third is the same-window claim
    itself, compared rather than assumed.
    """
    unmatched, null_count, disagreeing, rows, null_flag = con.execute("""
        SELECT count(*) FILTER (WHERE field_window_first_year IS NULL),
               count(*) FILTER (WHERE field_patents_in_window IS NULL),
               count(*) FILTER (WHERE field_window_first_year IS DISTINCT FROM
                                      window_first_year),
               count(*),
               count(*) FILTER (WHERE clears_floor IS NULL OR floor_patents IS NULL)
        FROM task3_candidates
    """).fetchone()

    if unmatched:
        raise ValueError(
            f"{unmatched:,} of {rows:,} candidate pair-years at {width} have no row in "
            f"02_field_years_{width}.parquet, so their field size is unknown and the tiny-field "
            f"floor would silently reject them. The two Task 2 tables were built from one "
            f"another, so this cannot happen unless they came from different runs. Rebuild with "
            f"python src/03_measure_dominance.py."
        )
    if null_count:
        raise ValueError(
            f"{null_count:,} of {rows:,} candidate pair-years at {width} have a NULL "
            f"field_patents_in_window. The floor test would be NULL, hence false, and every "
            f"with-exclusion cell would be understated while looking plausible. See the LEFT "
            f"JOIN in write_field_years() in src/03_measure_dominance.py."
        )
    if null_flag:
        raise ValueError(
            f"{null_flag:,} of {rows:,} candidate pair-years at {width} have a NULL "
            f"clears_floor or floor_patents. The stored floor flag would read as 'not floored' "
            f"in every with-floor cell, silently. Rebuild the field-year table with "
            f"python src/03_measure_dominance.py --force."
        )
    if disagreeing:
        raise ValueError(
            f"{disagreeing:,} of {rows:,} joined rows at {width} disagree about "
            f"window_first_year between the inventor table and the field-year table. The join "
            f"is on the window's END year, so this means the two tables do not mean the same "
            f"thing by a window, and Task 3's same-window requirement is not being enforced."
        )
    return {"width": width, "rows": rows,
            "detail": f"{rows:,} pair-years joined; 0 unmatched, 0 NULL field size, "
                      f"0 NULL floor flags, 0 window_first_year disagreements"}


def write_candidates(con, cfg, width):
    """Store the pool, so Task 4 does not have to re-read the 47-million-row build."""
    return io.write_parquet(
        con, con.sql("SELECT * FROM task3_candidates"), candidates_path(cfg, width), cfg,
        inputs=[inventor_field_years_path(cfg, width).as_posix(),
                field_years_path(cfg, width).as_posix()],
        step="04_select_dominant_inventors",
    )


def read_cached_candidates(con, cfg, width, funnel):
    """
    Reuse a stored pool instead of rebuilding it.

    Step 03 declares --force and never reads it, and nothing in src/ calls
    io.is_cached, so this is its first honest caller (rule 10). The funnel note
    matters: a run whose diary shows no filters at all would read as a run that
    dropped nothing, rather than as a run that skipped the work.
    """
    path = candidates_path(cfg, width)
    con.execute(f"CREATE OR REPLACE TEMP TABLE task3_candidates AS "
                f"SELECT * FROM read_parquet('{path.as_posix()}')")
    print(io.cache_message(path))
    funnel.note(f"{width}: the candidate pool was read from {path.name} rather than rebuilt, so "
                f"the two filters that produced it are recorded in the run that wrote it. Use "
                f"--force to rebuild.")
    return check_same_window_join(con, cfg, width)


# ---------------------------------------------------------------------------
# 4. Pair level. One table of qualifying pair-years; every cell is a filter on it.
# ---------------------------------------------------------------------------

def field_floor_flags_sql(cfg, width):
    """
    Per FIELD: does it ever clear the tiny-field floor inside the death window?

    This is the decomposition's C, and its denominator is the FIELD rather than the
    pair. That is the task list's own wording — "clearing the cutoff only in years
    when the field was tiny does not count, even if the field is large in other
    years" is a statement about the field. The pair-year reading is also not
    computable from the pool, because a pair-year in which the field is enormous
    and the inventor holds 0.3% sits below smallest_selectable_share(); getting it
    would mean a scan of the unfiltered 47-to-61-million-row table for a number
    nobody asked for.
    """
    return f"""
        SELECT
            field_id,
            bool_or({floor_clause(cfg)})              AS field_ever_clears_floor,
            count(*)                                  AS field_years_in_window,
            count(*) FILTER (WHERE {floor_clause(cfg)}) AS field_years_above_floor
        FROM read_parquet('{field_years_path(cfg, width).as_posix()}')
        WHERE year BETWEEN {cfg.deaths.year_min} AND {cfg.deaths.year_max}
        GROUP BY 1
    """


def rule_columns_sql(cfg):
    """
    One boolean column per column of the Task 3 table, from the tests in rules().

    A wide table of flags rather than a long table of (pair, rule, year): every
    cell is then `count(*) FILTER (WHERE <rule id>)` over one table, the rule's
    test appears exactly once in the whole step, and nothing has to concatenate
    SQL to add a rule.
    """
    return ",\n            ".join(f"({rule['test']}) AS {rule['id']}" for rule in rules(cfg))


def build_qualifying_years(con, cfg, width):
    """
    One row per (pair, window-year) inside the death window, carrying which rules
    that year clears and whether the field cleared the floor in that same year.

    THE SAME-WINDOW REQUIREMENT IS THIS TABLE'S SHAPE. `clears_floor` is the flag
    step 03 stored on that one field-year, so a cell that asks for the rule AND the
    floor is asking about one window-year, and a pair whose cutoff years and floor
    years never coincide cannot enter it.
    """
    con.execute(f"CREATE OR REPLACE TEMP TABLE task3_field_floor AS "
                f"{field_floor_flags_sql(cfg, width)}")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE task3_years AS
        SELECT
            c.inventor_id,
            c.field_id,
            c.year,
            c.share,
            c.rank,
            c.margin_uncapped,
            c.has_runner_up,
            c.margin_dense,
            c.has_dense_runner_up,
            c.n_tied_at_rank,
            c.inventor_patents_in_window,
            c.field_patents_in_window,
            c.inventors_in_field_window,
            c.floor_patents,
            c.clears_floor,
            f.field_ever_clears_floor,
            {rule_columns_sql(cfg)}
        FROM task3_candidates c
        JOIN task3_field_floor f ON f.field_id = c.field_id
        WHERE c.year BETWEEN {cfg.deaths.year_min} AND {cfg.deaths.year_max}
    """)


def build_career_shares(con, cfg, width):
    """
    Per counted pair: the median share over EVERY window-year it appears in the
    field, not only the qualifying ones.

    This is the durability statistic, and it replaces the year count for the reason
    §3.3 of the note sets out — persistence measured in overlapping windows cannot
    separate a plateau from a peak. A pair present in a field for thirteen
    window-years at 2.9% that pokes above 5% for three of them is not durably
    dominant, and no count of qualifying years can say so.

    It has to read the UNFILTERED panel: the pool keeps only rows above
    smallest_selectable_share(), so the low years — the ones that make a career
    median low — were dropped. A semi-join against the counted pairs keeps that to
    one streaming pass per width, reading four columns.

    Measured over dominance.first_compute_year through deaths.year_max, not the
    death window. Durability is a property of a career and the death window is a
    property of the DEATH data; those windows are complete from 1980, so extending
    left is free. Not extended right past deaths.year_max, because C-32 measured
    every window ending 2017-2022 as drawing on an incomplete filing year and grant
    lag is technology- and assignee-specific.
    """
    con.execute("""
        CREATE OR REPLACE TEMP TABLE task3_counted_pairs AS
        SELECT DISTINCT inventor_id, field_id FROM task3_years
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE task3_career AS
        SELECT
            i.inventor_id,
            i.field_id,
            count(*)          AS career_years,
            median(i.share)   AS career_median_share,
            max(i.share)      AS career_max_share
        FROM read_parquet('{inventor_field_years_path(cfg, width).as_posix()}') i
        SEMI JOIN task3_counted_pairs p
               ON p.inventor_id = i.inventor_id AND p.field_id = i.field_id
        WHERE i.year BETWEEN {cfg.dominance.first_compute_year} AND {cfg.deaths.year_max}
        GROUP BY 1, 2
    """)


# ---------------------------------------------------------------------------
# 5. The cells
# ---------------------------------------------------------------------------

def qualifying_clause(rule, with_floor):
    """What a pair-year must satisfy to be counted in one cell of one table."""
    return f"{rule['id']}" + (" AND clears_floor" if with_floor else "")


def cell(con, cfg, width, rule, with_floor):
    """
    One cell of the Task 3 table, and everything reported beside it.

    The counts the task list asks for are `pairs` and `inventors`. The rest is
    there because measuring the median showed the median cannot answer the question
    it was asked for — see the module docstring.

    `exposure` sums qualifying pair-years. Under Task 4's strict rule a pair is
    treated by a death in year t exactly when it qualifies in window-year t, so
    this counts treatment opportunities and the median does not.

    No p25/p75 anywhere. At cpc_class with the floor the 10% cell holds ONE pair,
    where quantile_cont returns p25 = p50 = p75 and prints as a tight distribution
    drawn from one observation. Median with n and range cannot do that, and the
    full sorted list of year counts goes into the CSV so no small-n threshold has
    to be invented.
    """
    keep = qualifying_clause(rule, with_floor)
    career_bar = rule["cutoff"] or cfg.dominance.alternative_rule.min_share
    row = con.execute(f"""
        WITH qualifying AS (
            SELECT * FROM task3_years WHERE {keep}
        ),
        pairs AS (
            SELECT
                q.inventor_id,
                q.field_id,
                count(*)                                  AS years,
                min(q.year)                               AS first_year,
                max(q.year)                               AS last_year,
                median(q.inventor_patents_in_window)      AS patents_behind,
                median(q.field_patents_in_window)         AS field_size,
                any_value(c.career_median_share)          AS career_median_share,
                any_value(c.career_years)                 AS career_years
            FROM qualifying q
            LEFT JOIN task3_career c
                   ON c.inventor_id = q.inventor_id AND c.field_id = q.field_id
            GROUP BY 1, 2
        )
        SELECT
            count(*)                                              AS pairs,
            count(DISTINCT inventor_id)                           AS inventors,
            coalesce(sum(years), 0)                               AS exposure,
            median(years)                                         AS median_years,
            min(years)                                            AS min_years,
            max(years)                                            AS max_years,
            -- A run with no gap. Reported for the rule-only and the rule-plus-floor
            -- populations separately, because with the floor in the same-window test
            -- a run can break purely because the FIELD dipped below the floor, and a
            -- single contiguity number could not be attributed to either.
            count(*) FILTER (WHERE last_year - first_year + 1 = years)  AS contiguous,
            count(*) FILTER (WHERE first_year = {cfg.deaths.year_min}
                                OR last_year  = {cfg.deaths.year_max})  AS edge_touching,
            median(years) FILTER (WHERE first_year > {cfg.deaths.year_min}
                                    AND last_year  < {cfg.deaths.year_max})
                                                                        AS median_years_interior,
            median(patents_behind)                                AS median_patents_behind,
            median(field_size)                                    AS median_counted_field_size,
            median(career_median_share)                           AS median_career_share,
            median(career_years)                                  AS median_career_years,
            -- "Durably dominant" in the only sense the data supports: the share
            -- sat above the bar for MOST of the pair's life in the field, not
            -- merely at its peak. The alternative rule's bar is its own share
            -- floor, since it has no cutoff of its own.
            count(*) FILTER (WHERE career_median_share >= {career_bar})
                                                                  AS career_above_cutoff,
            coalesce(list_sort(list(years)), [])                   AS year_counts
        FROM pairs
    """).fetchone()

    names = ["pairs", "inventors", "exposure", "median_years", "min_years", "max_years",
             "contiguous", "edge_touching", "median_years_interior", "median_patents_behind",
             "median_counted_field_size", "median_career_share", "median_career_years",
             "career_above_cutoff", "year_counts"]
    result = dict(zip(names, row))
    result.update({"width": width, "rule_id": rule["id"], "rule": rule["label"],
                   "kind": rule["kind"], "cutoff": rule["cutoff"], "with_floor": with_floor})
    result["loose_exposure"] = loose_exposure(con, cfg, rule, with_floor)
    return result


def loose_exposure(con, cfg, rule, with_floor):
    """
    Exposure under Task 4a's LOOSE timing rule, with no death data.

    A death in year d treats the pair under the loose rule if it qualified in any
    of the windows ending in d or in the {span - 1} years before it — that is, in
    any window-year s with d - {span} + 1 <= s <= d, which is the same as
    d in [s, s + {span} - 1]. So each qualifying window-year covers the next
    {span} death-years, and the union over a pair's qualifying years is what the
    loose rule would find.

    Built with the same `unnest(range(0, span))` expansion that
    window_rows_sql() in src/03_measure_dominance.py uses to lay a window over the
    filing years, which is not a coincidence: both turn one year into the span of
    years it reaches.

    DISTINCT is the whole point — overlapping runs must not be double-counted, and
    that is exactly what makes the ratio to strict exposure smaller than {span}.
    Clipped to the death window, because a death outside it is not usable.
    """
    span = cfg.windows.dominance_window_years
    keep = qualifying_clause(rule, with_floor)
    return con.execute(f"""
        SELECT count(*) FROM (
            SELECT DISTINCT q.inventor_id, q.field_id, q.year + step.ahead AS death_year
            FROM task3_years q,
                 (SELECT unnest(range(0, {span})) AS ahead) step
            WHERE {keep}
              AND q.year + step.ahead BETWEEN {cfg.deaths.year_min} AND {cfg.deaths.year_max}
        )
    """).fetchone()[0]


def decomposition(con, cfg, width, rule):
    """
    Exactly how the tiny-field floor meets the cutoff. The 2 August revision's
    request, and it is ONE new number rather than a third table.

    A is already the without-floor cell and B the with-floor cell; only C is new.
    The gap between A and B splits into two pathologies with different economics,
    which is why it is reported split:

      A - C   the field never clears the floor in ANY window in the death window.
              Permanently thin fields — the case the exclusion was written for.
      C - B   the pair cleared the cutoff, and the field cleared the floor, but
              never in the same window. Dominance during the field's thin youth:
              C-35's mechanism, and the one Task 7's matching will meet again.

    C is also the number that says how much the loose reading of the task list
    would overstate the count, which is why the sentence it was written to settle
    ("clearing the cutoff only in years when the field was tiny does not count,
    even if the field is large in other years") needed a measurement at all.
    """
    row = con.execute(f"""
        SELECT
            count(*)                                              AS pairs_a,
            count(*) FILTER (WHERE field_ever_clears_floor)        AS pairs_c,
            count(*) FILTER (WHERE ever_with_floor)                AS pairs_b
        FROM (
            SELECT inventor_id, field_id,
                   any_value(field_ever_clears_floor)              AS field_ever_clears_floor,
                   bool_or(clears_floor)                           AS ever_with_floor
            FROM task3_years
            WHERE {rule['id']}
            GROUP BY 1, 2
        )
    """).fetchone()
    a, c, b = row
    return {"width": width, "rule_id": rule["id"], "rule": rule["label"],
            "pairs_a_no_floor": a, "pairs_c_field_ever": c, "pairs_b_same_window": b,
            "lost_field_never_clears": a - c, "lost_never_together": c - b}


def alternative_rule_exclusions(con, cfg, width):
    """
    How many rank-1 pair-years the alternative rule drops for each of its two
    silent reasons, with and without the floor.

    Reported rather than relied on, because both are contingent on Task 2 storage
    decisions rather than on the rule: the tie is `rank_tie_rule: competition`
    giving a level top a margin of exactly 1.0, and the missing margin is
    `margin_when_no_runner_up: missing` making NULL >= 2 false. Under `cap` the
    lone inventors come back with the margin cap and pass, so the without-floor
    column of the alternative rule is not invariant to a key that has nothing to
    do with dominance.
    """
    rule = cfg.dominance.alternative_rule
    row = con.execute(f"""
        SELECT
            count(*)                                                       AS rank1_years,
            count(*) FILTER (WHERE margin_uncapped IS NULL)                AS no_runner_up,
            count(*) FILTER (WHERE margin_uncapped = 1.0)                  AS tied_top,
            count(*) FILTER (WHERE clears_floor)                           AS rank1_above_floor,
            count(*) FILTER (WHERE clears_floor AND margin_uncapped IS NULL)
                                                                           AS no_runner_up_above,
            count(*) FILTER (WHERE clears_floor AND margin_uncapped = 1.0) AS tied_top_above,
            count(*) FILTER (WHERE share >= {rule.min_share}
                                AND margin_uncapped < {rule.min_margin})   AS margin_too_small,
            -- Tied tops that fail `margin >= min_margin` ONLY because the top is
            -- tied: under the dense margin the co-leaders count as one and pass.
            count(*) FILTER (WHERE margin_uncapped = 1.0 AND share >= {rule.min_share}
                                AND margin_dense >= {rule.min_margin})     AS tied_top_rescued_by_dense,
            count(*) FILTER (WHERE clears_floor AND margin_uncapped = 1.0
                                AND share >= {rule.min_share}
                                AND margin_dense >= {rule.min_margin})     AS tied_top_rescued_above
        FROM task3_years
        WHERE rank <= {rule.max_rank}
    """).fetchone()
    names = ["rank1_years", "no_runner_up", "tied_top", "rank1_above_floor",
             "no_runner_up_above", "tied_top_above", "margin_too_small",
             "tied_top_rescued_by_dense", "tied_top_rescued_above"]
    return dict(zip(names, row), width=width)


def field_size_summary(con, cfg, width):
    """
    The "Send" line's median field size, in the two units a reader needs, plus the
    two that make a cutoff legible.

    `patents per year` is what the task list asks for and it is the window count
    divided by the window length. The window count itself is reported beside it
    because the floor is applied to THAT — quoting only the per-year figure would
    invite a reader to compare 68 against a floor of 100.

    The two extra columns exist because "5% of a field" is not a claim anyone can
    evaluate. 5% of a median main-group field is about one patent. The share of
    field-years clearing the floor is reported as a field-year share and as a
    PATENT share, because the field-year share overstates the loss wherever the
    discarded fields are small — which is the whole point of the floor.
    """
    row = con.execute(f"""
        SELECT
            count(*)                                              AS field_years,
            median(field_patents_in_window)                       AS median_patents_in_window,
            count(*) FILTER (WHERE {floor_clause(cfg)})           AS field_years_above_floor,
            median(field_patents_in_window) FILTER (WHERE {floor_clause(cfg)})
                                                                  AS median_above_floor,
            sum(field_patents_in_window)                          AS patents_all,
            sum(field_patents_in_window) FILTER (WHERE {floor_clause(cfg)})
                                                                  AS patents_above_floor,
            min(floor_patents)                                    AS floor_min,
            max(floor_patents)                                    AS floor_max
        FROM read_parquet('{field_years_path(cfg, width).as_posix()}')
        WHERE year BETWEEN {cfg.deaths.year_min} AND {cfg.deaths.year_max}
    """).fetchone()
    names = ["field_years", "median_patents_in_window", "field_years_above_floor",
             "median_above_floor", "patents_all", "patents_above_floor",
             "floor_min", "floor_max"]
    summary = dict(zip(names, row), width=width)
    span = cfg.windows.dominance_window_years
    summary["median_patents_per_year"] = summary["median_patents_in_window"] / span
    summary["cutoff_in_patents"] = summary["median_patents_in_window"] * cfg.dominance.cutoff
    return summary


def pairs_per_inventor(con, cfg, width, rule, with_floor):
    """
    How many fields one counted inventor dominates.

    Rule 11's motivating case — one person dominating three fields is one event,
    because inference clusters at the death. Reported once per width rather than as
    fifteen sets of parentheses that nearly all say 1.0, and the maximum is
    included because that is the number that decides whether the clustering
    matters at this width.
    """
    keep = qualifying_clause(rule, with_floor)
    row = con.execute(f"""
        SELECT count(*) FILTER (WHERE fields = 1),
               count(*) FILTER (WHERE fields = 2),
               count(*) FILTER (WHERE fields >= 3),
               coalesce(max(fields), 0)
        FROM (SELECT inventor_id, count(DISTINCT field_id) AS fields
              FROM task3_years WHERE {keep} GROUP BY 1)
    """).fetchone()
    return dict(zip(["one_field", "two_fields", "three_or_more", "max_fields"], row),
                width=width)


def fields_with_several_pairs(con, cfg, width, rule, with_floor):
    """
    Fields holding more than one counted dominant pair.

    Nobody asked for this and it bears directly on the identification. §5.2 of the
    design makes a treated field its own unit and matches it to control fields; a
    field with several dominant inventors is not several independent events, and it
    cannot serve as a clean control for itself. Task 5.5 asks the same question
    with deaths attached ("any field hit by more than one death within 10 years");
    the half of it that needs no death data is free here.

    C-30's finding that tied tops surviving into Task 3 are probably TEAMS is the
    same phenomenon seen from another side.
    """
    keep = qualifying_clause(rule, with_floor)
    row = con.execute(f"""
        SELECT count(*),
               count(*) FILTER (WHERE pairs > 1),
               coalesce(max(pairs), 0)
        FROM (SELECT field_id, count(DISTINCT inventor_id) AS pairs
              FROM task3_years WHERE {keep} GROUP BY 1)
    """).fetchone()
    return dict(zip(["fields", "fields_with_several", "max_pairs_in_a_field"], row),
                width=width)


def counted_inventors(con, cfg, rule, with_floor):
    """The set of inventors in one cell, for comparing widths against each other."""
    keep = qualifying_clause(rule, with_floor)
    frame = con.execute(f"SELECT DISTINCT inventor_id FROM task3_years "
                        f"WHERE {keep}").df()
    return set(frame.inventor_id)


def width_overlap(sets):
    """
    How many of one width's dominant inventors are also dominant at another.

    The three rows of the Task 3 table are three ways of cutting the same patents,
    never three samples, and this is the measurement that stops a reader treating
    them as three. Counts cannot be pooled either: item 9's describe_agreement
    learned that when pooling read 25.2% where the widths ran 45%, 38% and 23%.
    """
    names = list(sets)
    rows = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = len(sets[first] & sets[second])
            rows.append({"width": first, "other_width": second, "shared": shared,
                         "of_width": len(sets[first]), "of_other": len(sets[second])})
    return rows


def filing_years_behind_the_run(con, cfg, width, rule, with_floor):
    """
    Is a counted pair's run of qualifying window-years one burst of filing, or
    sustained activity?

    This is the check that corrects the task list's own premise, and it needs the
    FILING years rather than the window-years. Windows overlap by
    {span - 1} years, so a single heavy filing year occupies up to {span}
    consecutive window-years with no durability at all — which is why "median 3
    window-years" cannot be read as "mostly one-year spikes" OR as "durable".

    Only available where the field ids nest into 01_inventor_patents.parquet's
    frozen width: that file carries field_id at cfg.fields.strategy, and a broader
    CPC level is its prefix under cpc.assignment: primary. Returns None at any
    other width rather than implying coverage it does not have. C-39 D65 records
    the rejected alternative — the stored rolling sum can be differenced back into
    filing-year activity, which is exact but is a recursion with a guard, and this
    settles the same claim at two of three widths by reading the data directly.
    """
    frozen = cfg.fields.strategy
    if width == frozen:
        field_expression = "field_id"
    elif frozen == "cpc_subclass" and width == "cpc_class":
        field_expression = "left(field_id, 3)"
    else:
        return None

    span = cfg.windows.dominance_window_years
    keep = qualifying_clause(rule, with_floor)
    row = con.execute(f"""
        WITH qualifying AS (
            SELECT inventor_id, field_id, year FROM task3_years WHERE {keep}
        ),
        -- Every filing year the pair's qualifying windows cover. A window ending
        -- in t covers filings t-{span - 1} through t, so this is the window
        -- expansion of src/03_measure_dominance.py run backwards.
        covered AS (
            SELECT DISTINCT inventor_id, field_id, year - step.back AS filing_year
            FROM qualifying, (SELECT unnest(range(0, {span})) AS back) step
        ),
        activity AS (
            SELECT DISTINCT inventor_id, {field_expression} AS field_id, year AS filing_year
            FROM read_parquet('{io.interim_path(cfg, "01_inventor_patents.parquet").as_posix()}')
        ),
        per_pair AS (
            SELECT c.inventor_id, c.field_id, count(*) AS filing_years
            FROM covered c
            SEMI JOIN activity a
                   ON a.inventor_id = c.inventor_id
                  AND a.field_id = c.field_id
                  AND a.filing_year = c.filing_year
            GROUP BY 1, 2
        )
        SELECT count(*), median(filing_years),
               count(*) FILTER (WHERE filing_years = 1),
               count(*) FILTER (WHERE filing_years >= {span})
        FROM per_pair
    """).fetchone()
    return dict(zip(["pairs", "median_filing_years", "one_filing_year", "span_or_more"], row),
                width=width)


def pair_list_frame(con, cfg, width):
    """
    The counted pairs at the working cutoff with the floor, so a count has faces
    behind it and a reader who doubts a cell can open the rows.

    The working cutoff only. `dominance.cutoff` is a provisional value with no
    authority until Checkpoint 2 (its own comment in the config says so), and a
    file listing every pair at every cutoff would be four overlapping copies of
    the same rows.
    """
    keep = qualifying_clause(
        {"id": f"cutoff_{int(round(cfg.dominance.cutoff * 10000)):04d}"}, True)
    # Shares are printed rounded in the CSVs, and this is the reason rather than
    # tidiness. A window sum over a field-year is a float sum whose order DuckDB
    # does not fix, so its last digits are not reproducible across runs or thread
    # counts (C-37 D43, D48). Item 7 learned that a non-reproducible float in a
    # stamped output mints a new version of it on every run and destroys C-34's
    # guarantee. Item 9 rounds at the same key, so the two tables round alike.
    decimals = cfg.checkpoints.share_decimals
    frame = con.execute(f"""
        SELECT
            q.inventor_id,
            q.field_id,
            count(*)                                          AS years_above_cutoff,
            min(q.year)                                       AS first_year,
            max(q.year)                                       AS last_year,
            list_sort(list(q.year))                           AS years,
            round(max(q.share), {decimals})                    AS peak_share,
            round(any_value(c.career_median_share), {decimals})
                                                              AS career_median_share,
            any_value(c.career_years)                         AS career_years,
            median(q.inventor_patents_in_window)              AS median_patents_behind,
            median(q.field_patents_in_window)                 AS median_field_size,
            bool_or(q.rank = 1)                               AS ever_rank_1,
            -- The flag build_qualifying_years() already wrote, rather than the rule
            -- spelled out a second time: the CSV and the alternative-rule column
            -- have to be the same test or a reader comparing them finds a
            -- difference that is not in the data.
            bool_or(q.alternative)                            AS ever_clears_alternative,
            bool_or(q.alternative_dense)                      AS ever_clears_alternative_dense
        FROM task3_years q
        LEFT JOIN task3_career c
               ON c.inventor_id = q.inventor_id AND c.field_id = q.field_id
        WHERE {keep}
        GROUP BY 1, 2
        ORDER BY years_above_cutoff DESC, peak_share DESC, q.inventor_id
    """).df()
    frame["years"] = [compress_years(list(years)) for years in frame["years"]]
    return frame


# ---------------------------------------------------------------------------
# 6. The figure
# ---------------------------------------------------------------------------

def ecdf_tail(year_counts, max_k):
    """
    The share of counted pairs above the rule for at least k window-years, for
    k = 1 .. max_k.

    Pure Python on purpose: no DuckDB and no matplotlib, so the two properties
    that make the curve readable — it starts at 1.0 and never rises — can be
    tested directly rather than inferred from a picture.

    This is 1 - ECDF of a count, and it is deliberately not called a survival
    curve anywhere: "survival" invites a hazard reading of something that has no
    time dimension.
    """
    counts = list(year_counts)
    if not counts:
        return list(range(1, max_k + 1)), []
    return (list(range(1, max_k + 1)),
            [sum(1 for c in counts if c >= k) / len(counts) for k in range(1, max_k + 1)])


def floor_label(cfg, sizes):
    """
    The floor in words, across widths, read off what step 03 stored.

    One phrase when every width shares a number (absolute mode), otherwise the
    number per width — a 25th-percentile floor is 673 patents at cpc_class and
    7 at cpc_main_group, and a heading that said "p25" would hide exactly that.
    """
    numbers = {s["width"]: (s["floor_min"], s["floor_max"]) for s in sizes}
    if len(set(numbers.values())) == 1:
        low, high = next(iter(numbers.values()))
        return floor_words(cfg, low, high)
    per_width = ", ".join(
        f"{low:g}" + ("" if low == high else f"–{high:g}") + f" at {width}"
        for width, (low, high) in numbers.items())
    spec = (f"{cfg.dominance.floor_percentile:g}th percentile, {cfg.dominance.floor_scope}"
            if cfg.dominance.floor_mode == "percentile" else "absolute")
    return f"the floor of {per_width} patents ({spec})"


def persistence_figure(cfg, panels, max_k, label):
    """
    One panel per field width, one line per cutoff.

    Three things this figure has to carry or it misleads.

    n on every line. At cpc_class with the floor the four cutoffs hold about
    1,300, 120, 11 and 1 pairs; four curves drawn without their counts would
    invite a comparison between a distribution and a single observation.

    Two reference lines, from windows.dominance_window_years rather than typed.
    Consecutive windows share {span - 1} filing years, so ONE year of filing
    occupies {span} window-years — that line is the footprint of a single burst,
    not a durability threshold. The second, at 2 * {span} - 1, is the footprint of
    {span} consecutive filing years of activity, which is what durable dominance
    would actually look like in this unit.

    And a caption saying the populations are NESTED: every pair above 10% is above
    5%, so the lines are a subset comparison and not four samples.
    """
    span = cfg.windows.dominance_window_years
    widths = list(panels)
    fig, axes = plt.subplots(1, len(widths), figsize=(4.6 * len(widths), 4.4),
                            sharey=True)
    axes = axes if len(widths) > 1 else [axes]

    for ax, width in zip(axes, widths):
        for entry, colour in zip(panels[width], CUTOFF_COLOURS):
            ks, shares = ecdf_tail(entry["year_counts"], max_k)
            if not shares:
                continue
            ax.plot(ks, shares, color=colour, linewidth=1.8, marker="o", markersize=3,
                    label=f"{entry['cutoff']:.1%} (n={entry['pairs']:,})")

        ax.axvline(span, color=REFERENCE_COLOUR, linestyle=":", linewidth=1)
        ax.axvline(2 * span - 1, color=ALTERNATIVE_COLOUR, linestyle=":", linewidth=1)
        ax.set_xlabel("window-years above the cutoff (at least)")
        ax.set_title(f"{width}", fontsize=10)
        ax.set_ylim(0, 1.02)
        # The x-axis is logarithmic, and it is 02b's argument for its log y-axis
        # one axis over: the distribution runs to nearly thirty window-years while
        # every curve has collapsed by ten, so on a linear axis the region the
        # figure exists to show — between one window-year and the two reference
        # lines — occupies a third of the panel and the rest is empty. Truncating
        # instead would need a cut point that is not in the config and would hide
        # the handful of genuinely long runs, which are the interesting tail.
        ax.set_xscale("log")
        ax.set_xlim(1, max_k)
        ax.set_xticks([1, 2, 3, span, 2 * span - 1, 15, max_k])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("share of counted pairs")
    warning = (f"  —  SAMPLE MODE, CPC section {cfg.sample_cpc_section} only"
               if cfg.sample_mode else "")
    fig.suptitle(
        f"How long dominance lasts, among the pairs Task 3 counts{warning}\n"
        f"With {label}, window-years "
        f"{cfg.deaths.year_min}–{cfg.deaths.year_max}",
        fontsize=11)
    caption = (
        f"Dotted red at {span}: windows overlap by {span - 1} years, so ONE year of filing "
        f"occupies {span} window-years — this is the footprint of a single burst, not a "
        f"durability threshold.\n"
        f"Dotted grey at {2 * span - 1}: the footprint of {span} consecutive filing years, which "
        f"is what durable dominance looks like in this unit. The cutoff populations are NESTED "
        f"(every pair above the highest cutoff is above the lowest), so the lines compare "
        f"subsets, not samples."
    )
    fig.text(0.5, -0.02, caption, ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# 7. Validation — arithmetic, so it raises rather than reports
# ---------------------------------------------------------------------------

def check_pool_threshold_is_not_binding(con, cfg, width):
    """
    Prove that dropping rows below smallest_selectable_share() changed no count.

    The pool is a filter on the stored sample and not a discretisation only if a
    dropped row could never have entered a cell. That is an argument; this is the
    measurement. Every rule is evaluated on the rows just BELOW the threshold, read
    from the untruncated panel, and any that qualifies is a defect.
    """
    tests = " OR ".join(f"({rule['test']})" for rule in rules(cfg))
    smallest = smallest_selectable_share(cfg)
    qualifying, below = con.execute(f"""
        SELECT count(*) FILTER (WHERE {tests}), count(*)
        FROM read_parquet('{inventor_field_years_path(cfg, width).as_posix()}')
        WHERE year BETWEEN {cfg.deaths.year_min} AND {cfg.deaths.year_max}
          AND share < {smallest}
    """).fetchone()
    if qualifying:
        raise ValueError(
            f"{qualifying:,} of {below:,} pair-years at {width} sit below the pool threshold of "
            f"{smallest:.1%} and yet clear one of Task 3's rules, so dropping them changed a "
            f"count. smallest_selectable_share() is not the smallest share any rule selects on — "
            f"check dominance.cutoffs and dominance.alternative_rule.min_share."
        )
    return {"width": width, "below": below,
            "detail": f"0 of {below:,} pair-years below {smallest:.1%} clear any rule"}


def validate(cfg, width, cells, decomp, join_note, pool_note):
    """
    Everything that must hold by arithmetic, checked against the numbers actually
    reported. Returns (name, passed, detail); a failure stops the run after the
    files are written, as step 03 does.
    """
    results = [
        ("the same-window join is sound", True, join_note["detail"]),
        ("the candidate pool threshold changed no count", True, pool_note["detail"]),
    ]

    by_key = {(c["rule_id"], c["with_floor"]): c for c in cells}

    # B <= C <= A. The decomposition is three counts of nested populations, so any
    # other ordering means one of them is not counting what its name says.
    bad = [d for d in decomp
           if not (d["pairs_b_same_window"] <= d["pairs_c_field_ever"] <= d["pairs_a_no_floor"])]
    results.append((
        "the floor decomposition is nested (B <= C <= A)",
        not bad,
        f"{len(bad)} of {len(decomp)} rules out of order"
        if bad else f"all {len(decomp)} rules ordered",
    ))

    # A and B must be the two cells already reported, or the decomposition is
    # describing a different population from the tables.
    mismatched = []
    for d in decomp:
        for key, floor_setting in (("pairs_a_no_floor", False), ("pairs_b_same_window", True)):
            reported = by_key.get((d["rule_id"], floor_setting))
            if reported is not None and reported["pairs"] != d[key]:
                mismatched.append((d["rule_id"], floor_setting))
    results.append((
        "the decomposition's A and B are the table's own cells",
        not mismatched,
        f"{len(mismatched)} disagreements" if mismatched else "every cell reconciles",
    ))

    # A higher cutoff selects a subset of a lower one, so the counts cannot rise.
    rising = []
    for with_floor in {c["with_floor"] for c in cells}:
        ordered = [by_key[(f"cutoff_{int(round(c * 10000)):04d}", with_floor)]
                   for c in sorted_cutoffs(cfg)
                   if (f"cutoff_{int(round(c * 10000)):04d}", with_floor) in by_key]
        for lower, higher in zip(ordered, ordered[1:]):
            if higher["pairs"] > lower["pairs"] or higher["inventors"] > lower["inventors"]:
                rising.append((lower["rule"], higher["rule"], with_floor))
    results.append((
        "counts fall as the cutoff rises",
        not rising,
        f"{len(rising)} cutoff pairs where the count rose" if rising else "monotone at every cutoff",
    ))

    # A pair is an inventor and a field, so there cannot be more inventors than pairs.
    results.append((
        "pairs >= distinct inventors in every cell",
        all(c["pairs"] >= c["inventors"] for c in cells),
        f"{sum(1 for c in cells if c['pairs'] < c['inventors'])} cells with more inventors "
        f"than pairs",
    ))

    # Exposure sums qualifying years over the counted pairs, so it cannot be below
    # the pair count, and the loose rule cannot find fewer death-years than the
    # strict one.
    span = cfg.windows.dominance_window_years
    thin = [c for c in cells if c["exposure"] < c["pairs"]]
    loose_bad = [c for c in cells
                 if c["loose_exposure"] < c["exposure"]
                 or c["loose_exposure"] > span * c["exposure"]]
    results.append((
        "exposure >= pairs, and strict <= loose <= span x strict",
        not thin and not loose_bad,
        f"{len(thin)} cells with exposure below the pair count, {len(loose_bad)} with a loose "
        f"total outside [strict, {span} x strict]"
        if (thin or loose_bad) else f"all {len(cells)} cells within bounds",
    ))

    # A counted pair qualified in at least one year and cannot have qualified in
    # more years than the death window holds.
    window_years = cfg.deaths.year_max - cfg.deaths.year_min + 1
    out_of_range = [c for c in cells if c["pairs"]
                    and not (1 <= c["min_years"] <= c["max_years"] <= window_years)]
    results.append((
        f"every counted pair spends 1 to {window_years} window-years above its rule",
        not out_of_range,
        f"{len(out_of_range)} cells with a year count outside the range"
        if out_of_range else "every cell inside the range",
    ))

    return results


# ---------------------------------------------------------------------------
# 8. Frames and markdown
# ---------------------------------------------------------------------------

def cells_frame(cells, cfg):
    """
    The long-format CSV: one row per width x rule x floor setting.

    The full sorted list of year counts is a column, which is what makes the note
    able to print a median with n and a range instead of quantiles it cannot
    support at n = 1.
    """
    frame = pd.DataFrame(cells)
    frame["year_counts"] = [" ".join(str(y) for y in list(counts or []))
                            for counts in frame["year_counts"]]
    # Rounded rather than written out, because a share is a float sum whose last
    # digits are not reproducible across runs; an unrounded one in a stamped CSV
    # mints a new version of it on every run (C-37 D48, and item 7's lesson).
    decimals = cfg.checkpoints.share_decimals
    frame["median_career_share"] = [None if s is None else round(s, decimals)
                                    for s in frame["median_career_share"]]
    ordered = ["width", "rule_id", "rule", "kind", "cutoff", "with_floor", "pairs", "inventors",
               "exposure", "loose_exposure", "median_years", "min_years", "max_years",
               "median_years_interior", "contiguous", "edge_touching", "median_patents_behind",
               "median_counted_field_size", "median_career_share", "median_career_years",
               "career_above_cutoff", "year_counts"]
    return frame[ordered]


def decomposition_frame(decomp):
    return pd.DataFrame(decomp)[
        ["width", "rule_id", "rule", "pairs_a_no_floor", "lost_field_never_clears",
         "pairs_c_field_ever", "lost_never_together", "pairs_b_same_window"]]


def field_size_frame(sizes, cfg):
    frame = pd.DataFrame(sizes)
    frame["window_length_years"] = cfg.windows.dominance_window_years
    frame["working_cutoff"] = cfg.dominance.cutoff
    return frame[["width", "field_years", "median_patents_in_window",
                  "median_patents_per_year", "window_length_years", "field_years_above_floor",
                  "median_above_floor", "patents_all", "patents_above_floor",
                  "working_cutoff", "cutoff_in_patents"]]


def number(value, fmt="{:,.0f}"):
    """An em dash for a missing value, so an empty cell cannot read as a zero."""
    return "—" if value is None else fmt.format(value)


def cell_table_markdown(cfg, cells, with_floor):
    """
    The table the task list asks for: rows are widths, columns are the cutoffs and
    then the alternative rule, cells are pairs with distinct inventors beside them.

    Both numbers are stated in a fixed order announced once, rather than one of
    them being in brackets. The task list asks for pairs with inventors in
    parentheses; Task 4a asks for inventors with pairs in parentheses; and
    Checkpoint 2's viability guide is in inventors. Two tables read side by side
    with inverted brackets get misread, and inventors go first here because that is
    the unit the checkpoint decides on.
    """
    rule_list = rules(cfg)
    lines = [
        "Each cell is **distinct inventors / inventor × field pairs**, in that order. One "
        "person dominating three fields is three pairs and one event, because inference "
        "clusters at the death.",
        "",
        "| width | " + " | ".join(rule["label"] for rule in rule_list) + " |",
        "| --- | " + " | ".join("---:" for _ in rule_list) + " |",
    ]
    by_key = {(c["width"], c["rule_id"]): c for c in cells if c["with_floor"] == with_floor}
    for width in cfg.task2.widths:
        row = [f"| `{width}`"]
        for rule in rule_list:
            found = by_key.get((width, rule["id"]))
            row.append(f"{found['inventors']:,} / {found['pairs']:,}" if found else "—")
        lines.append(" | ".join(row) + " |")
    return lines


def statistics_table_markdown(cfg, cells, with_floor):
    """The per-cell statistics the counts alone cannot carry."""
    lines = [
        "| width | rule | inventors | pairs | exposure | loose | x | median yrs (n, range) | "
        "career share | patents behind | field size |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in cells:
        if c["with_floor"] != with_floor:
            continue
        multiplier = (f"{c['loose_exposure'] / c['exposure']:.2f}" if c["exposure"] else "—")
        spread = (f"{number(c['median_years'], '{:.0f}')} "
                  f"(n={c['pairs']:,}, {number(c['min_years'])}–{number(c['max_years'])})"
                  if c["pairs"] else "—")
        lines.append(
            f"| `{c['width']}` | {c['rule']} | {c['inventors']:,} | {c['pairs']:,} | "
            f"{c['exposure']:,} | {c['loose_exposure']:,} | {multiplier} | {spread} | "
            f"{number(c['median_career_share'], '{:.2%}')} | "
            f"{number(c['median_patents_behind'], '{:.0f}')} | "
            f"{number(c['median_counted_field_size'])} |"
        )
    return lines


def describe_persistence(cfg, cells, filing):
    """
    What the median number of years does and does not measure.

    Generated from this run's numbers with reachable branches, following
    item1.describe_floor_result and item7.describe_coverage_result: a different
    width, floor or cutoff could reverse every clause below, and a note whose prose
    asserted the 2026 answer would then be wrong while its tables were right.
    """
    span = cfg.windows.dominance_window_years
    working = f"cutoff_{int(round(cfg.dominance.cutoff * 10000)):04d}"
    counted = [c for c in cells if c["with_floor"] and c["rule_id"] == working and c["pairs"]]
    if not counted:
        return (f"No pair clears the working cutoff of {cfg.dominance.cutoff:.1%} with the floor "
                f"at any width, so there is no persistence to describe.")

    medians = [c["median_years"] for c in counted if c["median_years"] is not None]
    lowest, highest = min(medians), max(medians)
    spread = (f"{lowest:.0f}" if lowest == highest else f"{lowest:.0f} to {highest:.0f}")
    verdict = ("BELOW" if highest < span else "at or above")

    sentence = (
        f"At the working cutoff of {cfg.dominance.cutoff:.1%} with the floor, the median counted "
        f"pair is above the cutoff for {spread} window-years. Read that against {span}: "
        f"consecutive windows share {span - 1} filing years, so a single heavy filing year "
        f"occupies {span} consecutive window-years on its own. The median is therefore {verdict} "
        f"the footprint of one burst of filing"
    )
    if highest < span:
        sentence += (
            ", which means these pairs are not sitting on a plateau — the share is crossing the "
            "cutoff and falling back. That is the opposite of the reassuring reading, and it is "
            "what makes the career-share column rather than the year count the durability "
            "statistic here."
        )
    else:
        sentence += (
            ", so the year count cannot separate a durable position from one year of heavy "
            "filing, and the career-share column is what distinguishes them."
        )

    measured = [f for f in filing.values() if f and f["pairs"]]
    if measured:
        widths = " and ".join(f"`{f['width']}`" for f in measured)
        # Quoted from the width with the MOST counted pairs, not the width with the
        # highest single-filing-year rate. The rate is a proportion, and the worst
        # of three proportions is the one measured on the fewest pairs — at
        # cpc_class with the floor that is eleven, where one pair moves the figure
        # by nine points.
        largest = max(measured, key=lambda f: f["pairs"])
        rates = [f["one_filing_year"] / f["pairs"] for f in measured]
        spread = (f"{min(rates):.1%}" if len(rates) == 1
                  else f"{min(rates):.1%} to {max(rates):.1%} across those widths")
        sentence += (
            f" And they are not spikes either, which is measurable at {widths}, where the field "
            f"ids nest into Task 1's filing-year spine. At `{largest['width']}`, on "
            f"{largest['pairs']:,} counted pairs, the median pair filed in "
            f"{number(largest['median_filing_years'], '{:.0f}')} distinct years inside its "
            f"qualifying windows and {largest['one_filing_year'] / largest['pairs']:.1%} have a "
            f"single filing year behind the whole run ({spread}). They file steadily; what moves "
            f"is the denominator underneath them."
        )
    return sentence


def describe_floor_interaction(cfg, decomp):
    """Which of the two pathologies the floor is actually removing."""
    working = f"cutoff_{int(round(cfg.dominance.cutoff * 10000)):04d}"
    rows = [d for d in decomp if d["rule_id"] == working and d["pairs_a_no_floor"]]
    if not rows:
        return (f"No pair clears the working cutoff of {cfg.dominance.cutoff:.1%} at any width, "
                f"so the floor has nothing to interact with.")

    thin = sum(d["lost_field_never_clears"] for d in rows)
    untogether = sum(d["lost_never_together"] for d in rows)
    total = sum(d["pairs_a_no_floor"] for d in rows)
    kept = sum(d["pairs_b_same_window"] for d in rows)

    if untogether == 0:
        second = (" No pair anywhere cleared the cutoff and the floor in different windows, so "
                  "the task list's same-window requirement is not binding at this cutoff — the "
                  "strict and loose readings agree.")
    else:
        # Stated as a factor rather than a percentage increase, because the number
        # is larger than the population it is added to and "overstates by 661%"
        # reads as a rounding error at first glance.
        second = (f" {untogether:,} pairs cleared both at some point but never in the SAME "
                  f"window, which is the reading the task list rules out: dominance during the "
                  f"field's thin youth. Counting them would take the population from "
                  f"{kept:,} pairs to {kept + untogether:,}, a factor of "
                  f"{(kept + untogether) / kept:.1f} — so the same-window requirement is not "
                  f"pedantry, it decides most of the answer.")

    return (f"At {cfg.dominance.cutoff:.1%}, summed over the three widths, {total:,} pairs clear "
            f"the cutoff with the floor ignored and {kept:,} survive it. The loss splits into two "
            f"different things. {thin:,} pairs sit in a field that never clears the floor in any "
            f"window of the death window at all — permanently thin fields, the case the exclusion "
            f"was written for.{second}")


def describe_widths(cfg, sizes, overlap, cells):
    """Why the three rows are not three samples, in the numbers this run measured."""
    working = f"cutoff_{int(round(cfg.dominance.cutoff * 10000)):04d}"
    by_width = {c["width"]: c for c in cells
                if c["with_floor"] and c["rule_id"] == working}
    size_by_width = {s["width"]: s for s in sizes}
    # Broad and narrow are derived from the measured field size, not from the
    # position of a width in cfg.task2.widths: reordering that list must not be
    # able to make this sentence say the opposite of the table.
    ordered = sorted(size_by_width, key=lambda w: size_by_width[w]["median_patents_in_window"])
    narrow, broad = ordered[0], ordered[-1]

    shares = "; ".join(
        f"{row['shared']:,} of {row['of_width']:,} `{row['width']}` inventors are also dominant "
        f"at `{row['other_width']}`" for row in overlap)

    return (
        f"The three rows are three ways of drawing a box around the same patents, not three "
        f"samples, and the counts cannot be summed, averaged or pooled — item 9's "
        f"`describe_agreement` learned that when pooling read 25.2% where the widths ran 45%, 38% "
        f"and 23%. Counts rise as the field narrows for a mechanical reason: the median `{broad}` "
        f"field-year holds "
        f"{size_by_width[broad]['median_patents_in_window']:,.0f} patents over the window against "
        f"{size_by_width[narrow]['median_patents_in_window']:,.0f} at `{narrow}`, so the same "
        f"person's share is larger at `{narrow}` without anything about them having changed. At "
        f"the working cutoff of {cfg.dominance.cutoff:.1%} with the floor that is "
        f"{by_width[broad]['inventors']:,} inventors at `{broad}` against "
        f"{by_width[narrow]['inventors']:,} at `{narrow}`. They are also not the same people: "
        f"{shares}. So choosing a width chooses a population, not merely a threshold, and a "
        f"larger cell at a narrow width is not a larger sample of the same object."
    )


def read_interpretation(cfg, cells, decomp, sizes, overlap, several, per_inventor, filing):
    """
    The closing plain-English page: what a reader should look for.

    Generated from this run's numbers rather than typed, for the reason
    02b.read_interpretation gives, and it deliberately stops short of recommending
    a width or a cutoff. That is the Checkpoint 2 decision, and
    dominance.cutoff's own comment in the config says 0.05 has no authority until
    then.
    """
    working = f"cutoff_{int(round(cfg.dominance.cutoff * 10000)):04d}"
    size_by_width = {s["width"]: s for s in sizes}
    ordered = sorted(size_by_width, key=lambda w: size_by_width[w]["median_patents_in_window"])
    narrow = ordered[0]
    narrow_size = size_by_width[narrow]
    by_width = {c["width"]: c for c in cells if c["with_floor"] and c["rule_id"] == working}

    paragraphs = [
        f"**What a cutoff means in patents, before it means anything else.** A share is a ratio "
        f"and the tables above are full of them, so the first thing to look at is §4. The median "
        f"`{narrow}` field-year holds {narrow_size['median_patents_in_window']:,.0f} patents over "
        f"the {cfg.windows.dominance_window_years}-year window, which is "
        f"{narrow_size['median_patents_per_year']:,.1f} a year, so the working cutoff of "
        f"{cfg.dominance.cutoff:.1%} is about {narrow_size['cutoff_in_patents']:,.1f} patents "
        f"there. Among the pairs actually counted the median holds "
        f"{number(by_width[narrow]['median_patents_behind'], '{:.0f}')} patents in a field of "
        f"{number(by_width[narrow]['median_counted_field_size'])} — a claim about a person that "
        f"can be evaluated, which \"{cfg.dominance.cutoff:.0%} of a field\" is not.",

        f"**Then the persistence column, and read it against "
        f"{cfg.windows.dominance_window_years}.** "
        + describe_persistence(cfg, cells, filing),

        f"**Then exposure, not the pair count, if you are sizing a treated sample.** Every cell "
        f"above is an upper bound on Task 4's treated count, because Task 4 requires dominance at "
        f"the death and not at some point. The exposure column is the number that maps onto it: "
        f"under the strict timing rule a pair is treated by a death in year t exactly when it "
        f"qualifies in window-year t, so exposure counts treatment opportunities. The `x` column "
        f"beside it is what the loose rule buys, computed here rather than guessed at Task 4a. "
        f"Deaths themselves are Task 4 and this step does not touch them.",

        f"**Then the floor.** " + describe_floor_interaction(cfg, decomp),

        f"**Then the three rows.** " + describe_widths(cfg, sizes, overlap, cells),

        f"**One thing nobody asked for, which bears on the identification.** At the working "
        f"cutoff with the floor, "
        + "; ".join(f"{row['fields_with_several']:,} of {row['fields']:,} `{row['width']}` fields "
                    f"hold more than one counted dominant pair (up to "
                    f"{row['max_pairs_in_a_field']:,})" for row in several)
        + f". A field with several dominant inventors is not several independent events, and it "
        f"cannot be a clean control for itself — §5.2 of the design matches treated fields to "
        f"control fields, and Task 5.5 asks the same question again with deaths attached. The "
        f"mirror image is small: "
        + "; ".join(f"at `{row['width']}` {row['two_fields'] + row['three_or_more']:,} counted "
                    f"inventors dominate more than one field, at most {row['max_fields']:,}"
                    for row in per_inventor)
        + ", so rule 11's clustering concern — one person dominating several fields is still one "
          "event — is real mainly at the narrow widths.",

        f"**What this does not settle.** Nothing here chooses a width or a cutoff. A narrow field "
        f"is closer to what an inventor works on and produces shares large enough to be called "
        f"dominance; a broad field is closer to what economists mean by a technology market and "
        f"produces almost none. Task 4 attaches the deaths, and the ratio between a cell above "
        f"and its treated count is what the exposure column is there to forecast. Choosing the "
        f"pair is the Checkpoint 2 decision.",
    ]
    return "\n\n".join(paragraphs)


# One function per section of the note, each returning a list of lines, which is
# how the Task 2b items split the same job (item6's private _*_section helpers).
# The alternative is one 270-line write_report, and the owner of this repository
# has to be able to find the paragraph that produced a sentence she disagrees with.


def header_lines(cfg, switch_sentence):
    span = cfg.windows.dominance_window_years
    lines = [
        "# Task 3 — how many dominant inventors are there?",
        "",
        f"Config `{cfg.name}`, hash `{cfg.hash}`. Written by `{THIS_STEP}`.",
        "",
    ]
    if cfg.sample_mode:
        lines += [f"> **SAMPLE MODE — CPC section {cfg.sample_cpc_section} only.** These numbers "
                  f"describe a fraction of the data and are not results.", ""]
    return lines + [
        f"A pair is one inventor and one field. It is counted if the inventor's share of that "
        f"field reached the cutoff in at least one {span}-year window ending in "
        f"{cfg.deaths.year_min}–{cfg.deaths.year_max}. Counting **{cfg.counting.method}**, CPC "
        f"assignment **{cfg.cpc.assignment}**, tie rule **{cfg.dominance.rank_tie_rule}**, "
        f"margins read **uncapped**. {switch_sentence}",
        "",
        f"**This step does not touch deaths.** Every cell below is an upper bound on Task 4's "
        f"treated sample, because Task 4 requires dominance *at the death*. The `exposure` and "
        f"`x` columns in §1.3 are what forecast that, with no death data.",
        "",
        f"The counting sensitivity is not repeated here. Task 2b item 9 already produced these "
        f"cells under both counting methods off one stored dataset — see "
        f"`task2b_item9_task3_sensitivity.csv` — and `tests/test_task3.py` pins this step's "
        f"numbers to that table, so the two are comparable.",
        "",
        "## 1. The counts",
        "",
    ]


def counts_section(cfg, cells, label):
    """§1 — the table the task list asks for, then what its cells cannot carry."""
    span = cfg.windows.dominance_window_years
    lines = []
    floor_settings = sorted({c["with_floor"] for c in cells}, reverse=True)
    for with_floor in floor_settings:
        if with_floor:
            lines += [
                f"### 1.1 With the tiny-field floor — {label}, "
                f"{'at or above' if cfg.dominance.min_field_patents_inclusive else 'above'} "
                f"which a field-year is kept",
                "",
                f"The cutoff and the floor must be cleared in the **same** window. This is the "
                f"candidate population: Task 4 inherits it.",
                "",
            ]
        else:
            lines += [
                "### 1.2 Without the floor — diagnostic, not a candidate sample",
                "",
                f"How much of a cutoff's population is an artefact of field thinness. Items 1, 2 "
                f"and 6 between them established that what the floor removes is one-inventor "
                f"field-years, newborn fields and thin fields generally, so this table measures "
                f"the floor rather than offering an alternative sample.",
                "",
            ]
        lines += cell_table_markdown(cfg, cells, with_floor)
        lines += [""]

    lines += [
        "### 1.3 What the counts alone cannot carry",
        "",
        f"`exposure` is the number of qualifying (pair, window-year) observations; `loose` is the "
        f"same under Task 4a's loose timing rule (dominance in any of the {span} windows ending "
        f"in the death year or the {span - 1} before it) and `x` their ratio. `median yrs` is the "
        f"task list's median number of years above the cutoff, with n and range rather than "
        f"quartiles — at n = 1 a quartile spread is one observation wearing a distribution. "
        f"`career share` is the median share over **every** window-year the pair appears in the "
        f"field, from {cfg.dominance.first_compute_year}; `patents behind` and `field size` are "
        f"medians over the qualifying years.",
        "",
    ]
    for with_floor in floor_settings:
        lines += [f"**{'With' if with_floor else 'Without'} the floor.**", ""]
        lines += statistics_table_markdown(cfg, cells, with_floor)
        lines += [""]
    return lines


def floor_section(cfg, decomp):
    """§2 — the 2 August revision's request: exactly how the floor meets the cutoff."""
    lines = [
        "## 2. Exactly how the floor meets the cutoff",
        "",
        "One new number rather than a third table: **A** is §1.2's cell and **B** is §1.1's. "
        "**C** counts pairs clearing the rule in some window whose *field* clears the floor in "
        "some window — a property of the field, which is the task list's own wording (\"even if "
        "the field is large in other years\"). `B <= C <= A`, and the gap splits into two "
        "different pathologies.",
        "",
        "| width | rule | A: no floor | − field never clears | C: field ever clears | "
        "− never together | B: same window |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in decomp:
        lines.append(
            f"| `{d['width']}` | {d['rule']} | {d['pairs_a_no_floor']:,} | "
            f"{d['lost_field_never_clears']:,} | {d['pairs_c_field_ever']:,} | "
            f"{d['lost_never_together']:,} | {d['pairs_b_same_window']:,} |")
    return lines + ["", describe_floor_interaction(cfg, decomp), ""]


def persistence_section(cfg, cells, filing, figure_names):
    """§3 — how long dominance lasts, the figure, the censoring, and the spike check."""
    span = cfg.windows.dominance_window_years
    lines = [
        f"## 3. How long dominance lasts",
        "",
        describe_persistence(cfg, cells, filing),
        "",
        f"### 3.1 The figure",
        "",
    ]
    lines += [f"- `{name}`" for name in figure_names]
    lines += [
        "",
        f"One panel per width, one line per cutoff, on the with-floor population. The dotted red "
        f"line at {span} window-years is the footprint of a **single** year of filing, because "
        f"windows overlap by {span - 1} years; the grey line at {2 * span - 1} is the footprint "
        f"of {span} consecutive filing years. n is on every line, and the cutoff populations are "
        f"nested, so the lines compare subsets rather than samples.",
        "",
        "### 3.2 Where the death window cuts the runs",
        "",
        f"A run touching {cfg.deaths.year_min} or {cfg.deaths.year_max} is censored by the window "
        f"rather than ended by the inventor, so the median for the interior pairs is reported "
        f"beside the median for all of them. The censoring is part of the estimate — item 4's "
        f"lesson (C-32) on a different axis.",
        "",
        "| width | rule | pairs | touching an edge | median yrs, all | median yrs, interior |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in cells:
        if not c["with_floor"] or not c["pairs"]:
            continue
        lines.append(
            f"| `{c['width']}` | {c['rule']} | {c['pairs']:,} | "
            f"{c['edge_touching']:,} ({c['edge_touching'] / c['pairs']:.1%}) | "
            f"{number(c['median_years'], '{:.0f}')} | "
            f"{number(c['median_years_interior'], '{:.0f}')} |")

    lines += ["", "### 3.3 Are these one-year spikes?", ""]
    measured = {w: f for w, f in filing.items() if f}
    if measured:
        lines += [
            f"The window-year count cannot answer this, so it is measured against Task 1's "
            f"filing-year spine. Available only where the field ids nest into "
            f"`01_inventor_patents.parquet`'s frozen width (`{cfg.fields.strategy}`, and its CPC "
            f"prefix under `cpc.assignment: primary`); not derivable at the other widths, and "
            f"this table says so rather than implying coverage.",
            "",
            "| width | pairs | median distinct filing years | exactly 1 | "
            f"{span} or more |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for width in cfg.task2.widths:
            f = measured.get(width)
            if not f or not f["pairs"]:
                lines.append(f"| `{width}` | — | not derivable at this width | — | — |")
                continue
            lines.append(
                f"| `{width}` | {f['pairs']:,} | "
                f"{number(f['median_filing_years'], '{:.0f}')} | "
                f"{f['one_filing_year']:,} ({f['one_filing_year'] / f['pairs']:.1%}) | "
                f"{f['span_or_more']:,} ({f['span_or_more'] / f['pairs']:.1%}) |")
        lines += [""]
    return lines


def field_size_section(cfg, sizes):
    """§4 — the "Send" line's median field size, and what a cutoff is worth in patents."""
    span = cfg.windows.dominance_window_years
    lines = [
        "## 4. Field size, and what a cutoff is worth in patents",
        "",
        f"Median field size is what the \"Send\" line asks for. Both units are shown: patents per "
        f"year is the window count divided by {span}, and the window count itself is what the "
        f"floor is applied to. The last two columns are there because a share is not a claim "
        f"anyone can evaluate — the share of field-years clearing the floor is given beside the "
        f"share of PATENTS clearing it, because the field-year figure overstates the loss "
        f"wherever the discarded fields are small, which is the point of the floor.",
        "",
        "| width | field-years | median patents in window | per year | above the floor | "
        "median above floor | patents above floor | working cutoff in patents |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in sizes:
        lines.append(
            f"| `{s['width']}` | {s['field_years']:,} | "
            f"{s['median_patents_in_window']:,.0f} | {s['median_patents_per_year']:,.1f} | "
            f"{s['field_years_above_floor']:,} "
            f"({s['field_years_above_floor'] / s['field_years']:.1%}) | "
            f"{number(s['median_above_floor'])} | "
            f"{s['patents_above_floor'] / s['patents_all']:.1%} | "
            f"{s['cutoff_in_patents']:,.1f} |")
    return lines


def widths_section(cfg, sizes, overlap, several, per_inventor, cells):
    """§5 — why the three rows are alternatives and never strata."""
    lines = [
        "",
        "## 5. Three widths are not three samples",
        "",
        describe_widths(cfg, sizes, overlap, cells),
        "",
        "| width | also dominant at | shared inventors |",
        "| --- | --- | ---: |",
    ]
    for row in overlap:
        lines.append(f"| `{row['width']}` ({row['of_width']:,}) | `{row['other_width']}` "
                     f"({row['of_other']:,}) | {row['shared']:,} |")

    lines += [
        "",
        "| width | inventors in 1 field | 2 | 3+ | most fields one inventor holds | "
        "fields with >1 counted pair | most pairs in one field |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pi, sv in zip(per_inventor, several):
        lines.append(
            f"| `{pi['width']}` | {pi['one_field']:,} | {pi['two_fields']:,} | "
            f"{pi['three_or_more']:,} | {pi['max_fields']:,} | "
            f"{sv['fields_with_several']:,} of {sv['fields']:,} | "
            f"{sv['max_pairs_in_a_field']:,} |")
    return lines


def alternative_rule_section(cfg, exclusions):
    """§6 — the two groups the towering-figure rule drops without saying so."""
    rule = cfg.dominance.alternative_rule
    lines = [
        "",
        "## 6. What the alternative rule silently excludes",
        "",
        f"`rank {rule.max_rank}, margin >= {rule.min_margin:g}, share >= {rule.min_share:.1%}` "
        f"drops two groups for reasons that live in Task 2's storage rather than in the rule, so "
        f"they are counted here rather than relied on. A **tied** top has a margin of exactly 1.0 "
        f"under `rank_tie_rule: {cfg.dominance.rank_tie_rule}`, so ties fail — right, since if "
        f"two people are level neither towers. A rank-1 inventor who is the **only** inventor in "
        f"the field-window has no margin at all under "
        f"`margin_when_no_runner_up: {cfg.dominance.margin_when_no_runner_up}`, and `NULL >= "
        f"{rule.min_margin:g}` is false, so a person holding 100% of a field fails too. Also "
        f"right — nobody towers over an empty field — but it is contingent: under "
        f"`margin_when_no_runner_up: cap` those same monopolists return with the margin cap and "
        f"pass. Item 1 showed the floor removes every one-inventor field-year, which is why the "
        f"with-floor column is invariant to that key and the without-floor column is not.",
        "",
        f"The second margin definition (advisor, 2026-09-15) ranks DISTINCT shares, so tied "
        f"co-leaders count as one leader: `margin_dense` is the top share over the next "
        f"distinct share, and the `alternative_dense` column of §1 applies the same rule to it. "
        f"The last two columns count the tied tops the first definition drops ONLY because the "
        f"top is tied — rank 1, share >= {rule.min_share:.1%}, margin exactly 1.0 — that clear "
        f"`margin_dense >= {rule.min_margin:g}`; that is the whole difference between the two "
        f"columns of §1.",
        "",
        "| width | rank-1 pair-years | no runner-up | tied top | above the floor | "
        "no runner-up above | tied top above | tied top rescued by margin_dense | "
        "rescued above the floor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for e in exclusions:
        lines.append(
            f"| `{e['width']}` | {e['rank1_years']:,} | {e['no_runner_up']:,} | "
            f"{e['tied_top']:,} | {e['rank1_above_floor']:,} | {e['no_runner_up_above']:,} | "
            f"{e['tied_top_above']:,} | {e['tied_top_rescued_by_dense']:,} | "
            f"{e['tied_top_rescued_above']:,} |")
    return lines


def validation_section(results):
    """§7 — the guards, recorded in the note because a check nobody reads is not one."""
    lines = ["", "## 7. Validation", "", "| check | result | detail |", "| --- | :---: | --- |"]
    for width, checks in results.items():
        for name, passed, detail in checks:
            lines.append(f"| `{width}` — {name} | {'PASS' if passed else '**FAIL**'} | "
                         f"{detail} |")
    return lines


def closing_lines(cfg, cells, decomp, sizes, overlap, several, per_inventor, filing,
                  csv_paths, figure_names):
    """The reader's page, then what this run produced and what Task 4 inherits."""
    lines = ["", "---", "", "## What a reader should look for", "",
              read_interpretation(cfg, cells, decomp, sizes, overlap, several, per_inventor,
                                  filing),
              "", "---", "", "## Files from this run", ""]
    lines += [f"- `{Path(p).name}`" for p in csv_paths]
    lines += ["", "Figures:", ""] + [f"- `{name}`" for name in figure_names]
    lines += [
        "",
        f"`data/interim/03_dominance_candidates_<width>.parquet` holds the pair-years any Task 3 "
        f"or Task 4 rule can select, from {artifact_first_year(cfg)} to {cfg.deaths.year_max}, "
        f"with every continuous column intact. It starts four years before "
        f"{cfg.deaths.year_min} because Task 4a's loose rule reaches back that far; Task 3's own "
        f"cells use {cfg.deaths.year_min}–{cfg.deaths.year_max} only.",
        "",
    ]
    if loose_reach_is_clipped(cfg):
        lines += [
            f"> **The loose reach is clipped.** `dominance.first_compute_year` is "
            f"{cfg.dominance.first_compute_year}, which is later than "
            f"{cfg.deaths.year_min - (cfg.windows.dominance_window_years - 1)}, so Task 4a's "
            f"loose rule cannot see all {cfg.windows.dominance_window_years} windows for deaths "
            f"in the earliest years of the window. That is a limit of the stored panel, not of "
            f"this step.",
            "",
        ]
    return lines


def write_report(cfg, cells, decomp, sizes, overlap, several, per_inventor, exclusions,
                 filing, results, switch_sentence, figure_names, csv_paths):
    """Assemble the note out of its sections, in the order a reader meets them."""
    lines = (
        header_lines(cfg, switch_sentence)
        + counts_section(cfg, cells, floor_label(cfg, sizes))
        + floor_section(cfg, decomp)
        + persistence_section(cfg, cells, filing, figure_names)
        + field_size_section(cfg, sizes)
        + widths_section(cfg, sizes, overlap, several, per_inventor, cells)
        + alternative_rule_section(cfg, exclusions)
        + validation_section(results)
        + closing_lines(cfg, cells, decomp, sizes, overlap, several, per_inventor, filing,
                        csv_paths, figure_names)
    )
    return io.write_output_text(cfg, "task3_dominant_inventors.md", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the candidate pool, ignoring the cache")
    args = parser.parse_args()

    cfg = config_module.load(args.config)
    io.ensure_dirs(cfg)
    # Before anything is rewritten: say so if a report was edited by hand
    # since the last run. The stamped copy is authoritative, so this is a
    # warning and not an error (src/lib/io.py, and notes/decisions.md C-24).
    io.warn_about_hand_edited_outputs(cfg)
    Path(cfg.runtime.duckdb_temp_directory).mkdir(parents=True, exist_ok=True)

    funnel = Funnel("04_select_dominant_inventors", cfg)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{cfg.runtime.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory = '{cfg.runtime.duckdb_temp_directory}'")
    if cfg.runtime.duckdb_threads:
        con.execute(f"SET threads = {cfg.runtime.duckdb_threads}")

    print(f"Task 3 — config '{cfg.name}' (hash {cfg.hash}), "
          f"sample_mode={'ON: CPC section ' + cfg.sample_cpc_section if cfg.sample_mode else 'off'}")

    # The filenames in data/interim/ carry no config hash, so nothing about
    # 02_field_years_cpc_subclass.parquet says whether it holds one CPC section or
    # all of them. Reading a sample-mode table during a full run would produce a
    # complete, plausible Task 3 table for a thirteenth of the data.
    needed = [p for width in cfg.task2.widths
              for p in (inventor_field_years_path(cfg, width), field_years_path(cfg, width))]
    needed.append(io.interim_path(cfg, "01_inventor_patents.parquet"))
    io.require_built_by_this_config(needed, cfg,
                                    rebuild_with="python src/03_measure_dominance.py")

    check_metric_is_patent_share(cfg)
    check_alternative_rule_is_expressible(cfg)
    check_colours_cover_the_cutoffs(cfg)
    check_filing_years_are_reachable(cfg)
    floor_settings, switch_sentence = reporting_switches(cfg)

    cells, decomp, sizes, several, per_inventor, exclusions = [], [], [], [], [], []
    filing, results, panels, inventor_sets = {}, {}, {}, {}
    working = {"id": f"cutoff_{int(round(cfg.dominance.cutoff * 10000)):04d}"}
    csv_paths = []

    for width in cfg.task2.widths:
        print(f"\n--- {width} ---")
        inputs = [inventor_field_years_path(cfg, width), field_years_path(cfg, width)]
        if io.is_cached(candidates_path(cfg, width), inputs, cfg, force=args.force):
            join_note = read_cached_candidates(con, cfg, width, funnel)
        else:
            join_note = build_candidates(con, cfg, width, funnel)
            write_candidates(con, cfg, width)
        print(f"  join: {join_note['detail']}")

        build_qualifying_years(con, cfg, width)
        build_career_shares(con, cfg, width)
        pool_note = check_pool_threshold_is_not_binding(con, cfg, width)

        width_cells = [cell(con, cfg, width, rule, with_floor)
                       for with_floor in floor_settings
                       for rule in rules(cfg)]
        cells += width_cells
        decomp += [decomposition(con, cfg, width, rule) for rule in rules(cfg)]
        sizes.append(field_size_summary(con, cfg, width))
        exclusions.append(alternative_rule_exclusions(con, cfg, width))
        several.append(fields_with_several_pairs(con, cfg, width, working, True))
        per_inventor.append(pairs_per_inventor(con, cfg, width, working, True))
        filing[width] = filing_years_behind_the_run(con, cfg, width, working, True)
        inventor_sets[width] = counted_inventors(con, cfg, working, True)

        # The figure draws the with-floor population, because that is the one
        # Task 4 inherits. Only the share cutoffs: the alternative rule is not a
        # point on the same axis.
        panels[width] = [c for c in width_cells
                         if c["with_floor"] and c["kind"] == "cutoff"]

        results[width] = validate(
            cfg, width, width_cells,
            [d for d in decomp if d["width"] == width], join_note, pool_note)

        frame = pair_list_frame(con, cfg, width)
        stamped, _ = io.write_output_csv(
            cfg, f"task3_pairs_{width}.csv", frame, THIS_STEP)
        csv_paths.append(stamped)

        headline = next(c for c in width_cells
                        if c["with_floor"] == floor_settings[0] and c["rule_id"] == working["id"])
        print(f"  at {cfg.dominance.cutoff:.1%}"
              f"{' with the floor' if floor_settings[0] else ''}: "
              f"{headline['pairs']:,} pairs, {headline['inventors']:,} inventors, "
              f"exposure {headline['exposure']:,} pair-years")
        for name, passed, detail in results[width]:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")

    overlap = width_overlap(inventor_sets)

    max_k = max((c["max_years"] for c in cells if c["max_years"]), default=1)
    figure = persistence_figure(cfg, panels, max_k, floor_label(cfg, sizes))
    figure_stamped, _ = io.write_output_figure(cfg, figure, "task3_persistence.png",
                                              dpi=150, bbox_inches="tight")
    plt.close(figure)

    for name, frame in (("task3_cells.csv", cells_frame(cells, cfg)),
                        ("task3_decomposition.csv", decomposition_frame(decomp)),
                        ("task3_field_size.csv", field_size_frame(sizes, cfg))):
        stamped, _ = io.write_output_csv(cfg, name, frame, THIS_STEP)
        csv_paths.insert(0, stamped)

    stamped, plain = write_report(
        cfg, cells, decomp, sizes, overlap, several, per_inventor, exclusions, filing,
        results, switch_sentence, [figure_stamped.name], csv_paths)
    io.write_latest(cfg, "04_select_dominant_inventors", [stamped] + csv_paths + [figure_stamped],
                    summary="Task 3: dominant inventor x field pairs at four cutoffs and three "
                            "field widths, with and without the tiny-field floor.")
    funnel.finish()

    failures = [(w, n) for w, checks in results.items() for n, ok, _ in checks if not ok]
    if failures:
        raise SystemExit(
            f"\n{len(failures)} validation check(s) FAILED: "
            + "; ".join(f"{w}/{n}" for w, n in failures)
            + "\nThe Task 3 tables are on disk but must not be used."
        )
    print(f"\nAll checks passed. Report: {plain}")


if __name__ == "__main__":
    main()
