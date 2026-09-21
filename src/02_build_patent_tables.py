"""
Step 02 — Task 1: set up the data and fix the basic rules.

Reads : data/interim/*.parquet   (the converted PatentsView bulk tables)
        data/raw/kjl/*.csv       (ages, deaths, crosswalk — read, never modified)
Writes: data/interim/01_patents.parquet
        data/interim/01_patent_fields.parquet
        data/interim/01_inventor_patents.parquet     <- the spine of everything later
        data/interim/01_assignee_patents.parquet
        data/interim/01_citations.parquet
        data/interim/01_kjl_link.parquet
        data/interim/01_inventor_birth_years.parquet
        data/interim/01_inventor_deaths.parquet
        outputs/<config_name>/task1_setup_note.md
        logs/runs/<timestamp>_02_build_patent_tables/

Config keys used: patent_universe.*, citations.*, counting.*, cpc.*, fields.*,
deaths.*, birth_years.*, kjl_link.*, windows.*, sample_mode, sample_cpc_section,
columns.*, tables.*, paths.*, runtime.*

Task in the task list: Task 1. Implements the five rules the advisor fixes
"to use everywhere, from the start": application year rather than grant year;
fractional counting with a full-count alternative; the primary CPC code with an
all-codes alternative; deaths restricted to 1986-2016; and two birth-year lists,
a looser one and a stricter one.

On those two lists: the task list assigns the strict rule to the general pool
and the loose one to potential treated inventors. Read literally — "the two
files agree" means the ages file and the death file — the strict rule needs a
death record, so it has no analogue for a living inventor and cannot define a
control group. The general pool therefore uses Rule A (loose) necessarily, and
Rule B (strict) is a robustness restriction on the treated. That is a departure
from the task list's wording and is flagged in notes/decisions.md C-25.

Serves: every sub-question. This step defines the population that SQ1-SQ6 are
all measured on, which is why each of the five rules is a config switch rather
than a branch written into the code — changing any of them must never mean
editing Python.

What this step does NOT do: it computes no shares, ranks or margins. Dominance
is Task 2. Everything here is continuous or raw and stored unfiltered wherever
a cutoff would otherwise be applied early.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb

from src.lib import asof
from src.lib import config as config_module
from src.lib import fields as fields_module
from src.lib import io
from src.lib import patent_ids
from src.lib.funnel import Funnel


# How the tie in a death year is broken. The rule is chosen in the config; this
# is only the translation into an ordering, so that the four options cannot
# drift apart from the four SQL fragments that implement them.
TIE_ORDERINGS = {
    "modal_then_earliest": "records_with_this_year DESC, death_year ASC",
    "earliest": "death_year ASC",
    "latest": "death_year DESC",
    "drop": "death_year ASC",
}


def raw(cfg, key):
    return io.interim_path(cfg, cfg.tables[key]).as_posix()


def kjl(cfg, key):
    return io.raw_kjl_path(cfg, cfg.tables[key]).as_posix()


def out(cfg, name):
    return io.interim_path(cfg, name)


def quoted_list(values):
    """['utility'] -> "'utility'" — for an SQL IN clause."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def count_rows(con, path):
    return con.sql(f"SELECT count(*) FROM read_parquet('{Path(path).as_posix()}')").fetchone()[0]


# ---------------------------------------------------------------------------
# 1. Fields — built first, because sample mode enters here and everything
#    downstream inherits it through the patents that have a field at all.
# ---------------------------------------------------------------------------

def build_fields(con, cfg, funnel):
    """Assign every patent to a field, using the strategy named in the config."""
    relation = fields_module.assign_fields(con, cfg, funnel=funnel)
    relation.to_view("patent_fields_raw", replace=True)
    return relation


# ---------------------------------------------------------------------------
# 2. Patents — the universe, and the date rule
# ---------------------------------------------------------------------------

def build_patents(con, cfg, funnel):
    """
    One row per patent, with the rules of Task 1 applied and recorded.

    The date rule lives here: `year` is the column every later step uses, and it
    is the application year or the grant year according to patent_universe.date_by.
    Both are kept, so a robustness run does not have to rebuild anything.
    """
    columns = cfg.columns
    universe = cfg.patent_universe

    # The one place configured column names enter SQL. Everything after this
    # uses the canonical names on the left of each AS.
    reader = """
        SELECT
            p."{patent_id}"                                       AS patent_id,
            p."{patent_type}"                                     AS patent_type,
            COALESCE(TRY_CAST(p."{withdrawn}" AS INTEGER), 0)     AS withdrawn,
            TRY_CAST(p."{patent_date}" AS DATE)                   AS grant_date,
            TRY_CAST(a."{filing_date}" AS DATE)                   AS filing_date
        FROM read_parquet('{patents_path}') p
        LEFT JOIN read_parquet('{application_path}') a
               ON a."{patent_id}" = p."{patent_id}"
    """.format(
        patent_id=columns.patent_id, patent_type=columns.patent_type,
        withdrawn=columns.withdrawn, patent_date=columns.patent_date,
        filing_date=columns.filing_date,
        patents_path=raw(cfg, "patents"), application_path=raw(cfg, "application"),
    )
    # A withdrawn flag that is missing means "not marked withdrawn", which is
    # why COALESCE is used above and only there.
    patents = con.sql(reader).project(
        "patent_id, patent_type, withdrawn, grant_date, filing_date, "
        "year(grant_date) AS grant_year, year(filing_date) AS application_year"
    )

    date_column = "filing_date" if universe.date_by == "application" else "grant_date"
    year_column = "application_year" if universe.date_by == "application" else "grant_year"

    patents = funnel.filter(
        patents, "patent_type",
        f"patent_type IN ({quoted_list(universe.patent_types)})",
        why=(
            f"Only {', '.join(universe.patent_types)} patents are kept. Design and plant "
            f"patents are patents in law but are not inventions in the sense this design is "
            f"about, and reissues restate an existing patent rather than adding an invention, "
            f"so counting them would inflate both field sizes and individual output. This "
            f"also removes almost every patent that carries no CPC code at all: 100 per cent "
            f"of design and plant patents are unclassified, against 0.2 per cent of utility "
            f"patents."
        ),
    )

    if universe.drop_withdrawn:
        patents = funnel.filter(
            patents, "not_withdrawn", "withdrawn = 0",
            why=(
                "Withdrawn patents are removed. A withdrawal is an administrative act by the "
                "patent office, so the record no longer corresponds to a granted invention "
                "and should not count towards anyone's share of a field."
            ),
        )

    patents = funnel.filter(
        patents, "has_usable_date", f"{date_column} IS NOT NULL",
        why=(
            f"Every patent is dated by its {universe.date_by} date, following the advisor's "
            f"rule that patents are dated by when the inventive act happened rather than when "
            f"examination finished, because grant lag varies by technology and by decade. A "
            f"patent with no usable {universe.date_by} date cannot be placed in a year, and a "
            f"patent that cannot be placed in a year cannot enter a five-year window."
        ),
    )

    patents = funnel.filter(
        patents, "year_at_or_after_data_start", f"{year_column} >= {universe.first_year}",
        why=(
            f"Patents dated before {universe.first_year} are removed, because that is the "
            f"first filing year whose denominator is complete — not because the data begins "
            f"there. A grant date is always on or after the filing date, and this dataset "
            f"contains every patent granted from {universe.first_year} onwards, so a patent "
            f"filed in {universe.first_year} or later appears here if it was ever granted, "
            f"while one filed a year earlier may have been granted a year earlier and be "
            f"absent. Earlier filing years are therefore truncated on grant lag, and severely: "
            f"68,052 patents survive from filing year 1975 and 534 from 1970, against roughly "
            f"100,000 actually filed in each. Since a share is a ratio whose denominator must "
            f"be the whole field, what survives from those years would not give a noisy share "
            f"but one wrong by orders of magnitude, inflated for precisely those inventors "
            f"who happened to hold slow-granted patents. A small number of these rows are "
            f"also outright data errors: the file contains filing years such as 975, 1074 "
            f"and 1298."
        ),
    )

    if universe.drop_incomplete_filing_years:
        patents = funnel.filter(
            patents, "year_complete",
            f"{year_column} <= {universe.filings_complete_through}",
            why=(
                f"Patents dated after {universe.filings_complete_through} are removed. Only "
                f"granted patents appear in this data, and patents filed more recently are "
                f"still working through examination, so recent {universe.date_by} years are "
                f"truncated. Keeping them would show a collapse in inventive activity that is "
                f"an artefact of the data ending rather than anything real. Nothing in this "
                f"design needs them: the last usable death is {cfg.deaths.year_max}, and the "
                f"post-death window closes {cfg.windows.post_years} years later."
            ),
        )

    patents = funnel.filter(
        patents, "has_a_field",
        "patent_id IN (SELECT patent_id FROM patent_fields_raw)",
        why=(
            "Patents with no field are removed. A patent outside every field contributes to "
            "no denominator and can make nobody dominant, so it cannot enter this design. "
            + (f"Sample mode is on, so this is also where the analysis is cut down to CPC "
               f"section {cfg.sample_cpc_section}." if cfg.sample_mode else "")
        ),
    )

    patents = patents.project(
        f"patent_id, patent_type, grant_date, filing_date, application_year, grant_year, "
        f"{year_column} AS year"
    )
    patents.to_view("clean_patents", replace=True)
    return patents


# ---------------------------------------------------------------------------
# 3. The spine: inventor x patent, with the counting rule applied
# ---------------------------------------------------------------------------

def build_inventor_patents(con, cfg, funnel):
    """
    One row per inventor per patent per field, carrying the counting weight.

    This is the table every later step builds on, so the counting rule is
    applied exactly once, here. Under fractional counting a patent with n
    inventors contributes 1/n to each of them, which is what makes shares within
    a field sum to one and stops a person on large teams accumulating dominance
    simply by being one of many.
    """
    columns = cfg.columns
    reader = """
        SELECT
            "{patent_id}"                            AS patent_id,
            CAST("{inventor_sequence}" AS INTEGER)   AS inventor_sequence,
            "{inventor_id}"                          AS inventor_id
        FROM read_parquet('{path}')
    """.format(
        patent_id=columns.patent_id, inventor_sequence=columns.inventor_sequence,
        inventor_id=columns.inventor_id, path=raw(cfg, "inventors"),
    )
    links = con.sql(reader)

    links = funnel.filter(
        links, "inventors_of_clean_patents",
        "patent_id IN (SELECT patent_id FROM clean_patents)",
        why=(
            "Inventor records are restricted to the patents that survived the rules above. "
            "This drops no inventor on their own account — only their appearances on patents "
            "that are not part of this universe."
        ),
    )

    # Team size counts everyone named on the patent. The filter above removed
    # whole patents, never individual inventors from a surviving patent, so the
    # team is still intact at this point.
    #
    # `listings` counts how many times ONE person is named on ONE patent, and it
    # is not always 1: 1,779 patent-inventor pairs in this data name the same
    # disambiguated inventor two or three times, patent 6135931 naming
    # fl:wi_ln:padula-1 three times on a team of three. Both counts are taken
    # here, BEFORE the field join below, for the same reason: under
    # cpc.assignment: all that join repeats every row once per CPC code, and a
    # window function counted after it would multiply both by the number of
    # fields.
    links = links.project(
        "patent_id, inventor_sequence, inventor_id, "
        "count(*) OVER (PARTITION BY patent_id) AS team_size, "
        "count(*) OVER (PARTITION BY patent_id, inventor_id) AS listings"
    )

    # Fractional: the patent is worth 1 and is split across the names on it, so a
    # person named three times collects three thirds and ends with the whole
    # patent. That is right, and it is why the divisor is team_size.
    #
    # Full: the patent is worth 1 TO EACH INVENTOR — one patent, one credit,
    # however many times the office listed them. So the divisor is `listings`,
    # not 1. Writing 1.0 here credited the padula case with three patents and
    # over-credited 857 people across 5,621 inventor-field-year rows, 49 of them
    # rank-1 rows, by up to 61 patents. Nothing had run this branch, which is why
    # it survived; measured and fixed at Task 2b item 9 (notes/decisions.md C-38).
    weight = "1.0 / team_size" if cfg.counting.method == "fractional" else "1.0 / listings"
    per_field = ("count(*) OVER (PARTITION BY patent_id)"
                 if cfg.cpc.assignment == "all" else "1")

    con.sql(f"""
        SELECT patent_id, field_id, {per_field} AS fields_on_patent
        FROM patent_fields_raw
    """).to_view("fields_with_count", replace=True)

    links.to_view("inventor_links", replace=True)

    # Under the all-codes alternative a patent appears in several fields. Whether
    # that means it counts fully in each, or is split across them, is a config
    # choice: cpc.multi_code_weighting.
    field_divisor = ("fields_on_patent"
                     if (cfg.cpc.assignment == "all"
                         and cfg.cpc.multi_code_weighting == "fractional")
                     else "1")

    spine = con.sql(f"""
        SELECT
            l.patent_id,
            l.inventor_id,
            l.inventor_sequence,
            l.team_size,
            f.field_id,
            p.application_year,
            p.grant_year,
            p.year,
            ({weight}) / {field_divisor}   AS weight
        FROM inventor_links l
        JOIN clean_patents  p ON p.patent_id = l.patent_id
        JOIN fields_with_count f ON f.patent_id = l.patent_id
    """)

    funnel.checkpoint("inventor x patent x field spine", spine)
    _check_weights(con, spine, cfg)

    # The join above can quietly lose a patent that has no inventor record at
    # all. No row may disappear without being recorded, so it is counted and
    # reported rather than left to be noticed later.
    spine.to_view("spine_for_count", replace=True)
    orphans = con.sql("""
        SELECT count(*) FROM clean_patents
        WHERE patent_id NOT IN (SELECT patent_id FROM spine_for_count)
    """).fetchone()[0]
    if orphans:
        funnel.note(
            f"{orphans:,} patents carry no inventor record at all and therefore do not appear "
            f"in the inventor-level spine. They remain in the patents table, so they still "
            f"count towards the size of their field — a patent with an unnamed inventor is "
            f"still a patent in the field — but they can contribute to nobody's share."
        )
    return spine


def _check_weights(con, spine, cfg):
    """
    Fail loudly if the counting weights are not what they claim to be.

    Under fractional counting with primary codes, every patent's weights must
    add to exactly one. If that is ever false, every share computed afterwards
    is wrong in a way that looks entirely plausible.

    The gate on that check is load-bearing rather than an omission, and closing it
    would look like a tidy-up. Under FULL counting a patent's weights add to the
    number of DISTINCT inventors named on it — one per person, by construction —
    so 5,376,170 of 8,191,355 patents (65.6%) would fail.

    Task 2b item 8 measured that as 5,376,895 when the full branch still summed to
    the team SIZE. Item 9's fix moved it by 725: exactly the patents on which every
    listing is the same person, where the team size exceeds the head count of
    people. The verdict is unchanged and the gate stays shut either way. See
    notes/decisions.md D47 and C-38.
    """
    spine.to_view("spine_to_check", replace=True)
    bad = con.sql("""
        SELECT count(*) FROM spine_to_check WHERE weight <= 0 OR weight > 1
    """).fetchone()[0]
    if bad:
        raise ValueError(f"{bad:,} rows have a counting weight outside (0, 1].")

    if cfg.counting.method == "fractional" and cfg.cpc.assignment == "primary":
        off = con.sql("""
            SELECT count(*) FROM (
                SELECT patent_id, sum(weight) AS total
                FROM spine_to_check GROUP BY 1
                HAVING abs(sum(weight) - 1.0) > 1e-9
            )
        """).fetchone()[0]
        if off:
            raise ValueError(
                f"{off:,} patents have fractional weights that do not sum to 1. Shares "
                f"computed from this table would not be shares."
            )


# ---------------------------------------------------------------------------
# 4. Assignees and citations — organised now, used later
# ---------------------------------------------------------------------------

def build_assignee_patents(con, cfg, funnel):
    """Patent to assignee, for the employer measures in Task 5 and Task 9."""
    columns = cfg.columns
    reader = """
        SELECT
            "{patent_id}"                            AS patent_id,
            CAST("{assignee_sequence}" AS INTEGER)   AS assignee_sequence,
            "{assignee_id}"                          AS assignee_id,
            disambig_assignee_organization           AS assignee_organization
        FROM read_parquet('{path}')
    """.format(
        patent_id=columns.patent_id, assignee_sequence=columns.assignee_sequence,
        assignee_id=columns.assignee_id, path=raw(cfg, "assignees"),
    )
    return funnel.filter(
        con.sql(reader), "assignees_of_clean_patents",
        "patent_id IN (SELECT patent_id FROM clean_patents)",
        why=(
            "Assignee records are restricted to the patents in this universe. Note that "
            "assignee identifiers are not firm identifiers: matching assignees to firms is on "
            "the advisor's do-not-do-yet list until Checkpoint 2, so this table records who "
            "the patent was assigned to and nothing more."
        ),
    )


# "The records of this edge disagree about who supplied the reference" — two or
# more distinct categories on one (citing, cited) pair, ignoring the ones that
# are simply unrecorded.
#
# Written once here because it is asked three times below, and written with
# min/max rather than the count(DISTINCT raw_category) > 1 that says the same
# thing in plainer English, for a reason that is not style. Both ignore nulls,
# so they agree on every group: with no non-null category min and max are both
# null and IS DISTINCT FROM is false; with one they are equal; with two or more
# they differ. But DuckDB implements a DISTINCT aggregate by keeping a distinct
# table PER GROUP, in the one pool it cannot write out to the temp directory,
# and there are 121,477,265 groups here. That pool passed 800 MB of a 2 GB limit
# and was still climbing when step 02 died of it five times on 24 August 2026,
# while everything spillable sat at 1.1 GB. min and max hold two values each and
# left the pool flat at 363 MB, writing the same file in 216 seconds.
#
# tests/test_citation_spine.py asserts the plan contains no DISTINCT aggregate,
# and the conflict tests beside it pin that the answer did not change.
CATEGORIES_DISAGREE = "min(raw_category) IS DISTINCT FROM max(raw_category)"


# Whether a citation counts as examiner-found, when the raw file records the
# same (citing, cited) edge twice and the two records disagree. The rule is
# chosen in the config; this is only the translation into SQL, so that the three
# options cannot drift apart from the three aggregate expressions that implement
# them — the same arrangement as TIE_ORDERINGS above.
#
# Each expression is an aggregate over one (citing_patent_id, cited_patent_id)
# group and returns a nullable boolean. They differ for a pair recorded once as
# examiner-cited and once as applicant-cited: `any` calls it examiner, `all`
# calls it not-examiner, `drop_conflicts` refuses to say.            [C-46]
EXAMINER_FLAG_RULES = {
    "any_examiner": "bool_or(raw_category = 'cited by examiner')",
    "all_examiner": "bool_and(raw_category = 'cited by examiner')",
    "drop_conflicts": (
        f"CASE WHEN {CATEGORIES_DISAGREE} THEN NULL "
        "ELSE bool_or(raw_category = 'cited by examiner') END"
    ),
}


def build_citations(con, cfg, funnel):
    """
    Citations received by the patents in this universe.

    Organised on the CITED side: a row exists when the cited patent is one of
    ours, whoever did the citing. That is the shape a dependence measure needs,
    since it asks how much of a field's incoming attention goes to one person.

    The citing patent contributes BOTH of its dates, and which one is used is
    not a detail. A measure dated t may use a citation only if the citing patent
    was FILED by t, and the whole project dates by filing year
    (patent_universe.date_by). An earlier version of this function stored the
    grant year alone and its docstring claimed that was "what makes it possible
    later to count only citations that had been made by a given date" — which is
    wrong twice over: it is the wrong clock for this project, and no stored
    column can make that claim, because a reference list is not complete at
    filing. Applicants supplement it during prosecution and examiners add to it,
    typically one to four years later. The two dates therefore BRACKET the date
    a citation was actually made, and the gap between them is the one look-ahead
    a citation-based pre-death measure cannot remove. Keeping both is what makes
    that limitation measurable instead of invisible.                   [C-46]

    citation_category records who put the reference on the patent. It is the
    only column that can separate the inventor's own reliance on prior art from
    the examiner's reading of it, and it is empty for grant years through 2001.
    """
    columns = cfg.columns
    # LEFT JOINs on both sides on purpose: a citing patent outside this universe
    # keeps its row and gets a null filing year, which is exactly the marker a
    # field-side denominator needs in order to exclude it. An inner join would
    # drop those rows here, silently, and rule 6 forbids that.
    reader = """
        SELECT
            c."{patent_id}"                           AS citing_patent_id,
            c."{citation_patent_id}"                  AS cited_patent_id,
            TRY_CAST(g."{patent_date}" AS DATE)       AS citing_grant_date,
            year(TRY_CAST(g."{patent_date}" AS DATE)) AS citing_grant_year,
            u.application_year                        AS citing_application_year,
            nullif(trim(c."{citation_category}"), '') AS raw_category
        FROM read_parquet('{citations_path}') c
        LEFT JOIN read_parquet('{patents_path}') g
               ON g."{patent_id}" = c."{patent_id}"
        LEFT JOIN clean_patents u
               ON u.patent_id = c."{patent_id}"
    """.format(
        patent_id=columns.patent_id, citation_patent_id=columns.citation_patent_id,
        patent_date=columns.patent_date, citation_category=columns.citation_category,
        citations_path=raw(cfg, "citations"), patents_path=raw(cfg, "patents"),
    )
    received = funnel.filter(
        con.sql(reader), "citations_received_by_clean_patents",
        "cited_patent_id IN (SELECT patent_id FROM clean_patents)",
        why=(
            "Citations are kept where the CITED patent belongs to this universe, whatever the "
            "citing patent is. Restricting the citing side as well would discard attention "
            "arriving from outside the sample, which is real attention: a field can be built "
            "on from anywhere. A measure that needs the citing patent to be in the universe — "
            "as a field-level dependence measure does, because its denominator is the field's "
            "own patents — applies that restriction itself, where it can be counted against "
            "the measure rather than against the whole table."
        ),
    )
    if not cfg.citations.deduplicate_edges:
        funnel.note(
            "Duplicate (citing, cited) edges were NOT collapsed, because "
            "citations.deduplicate_edges is false. Any measure counting citation rows rather "
            "than distinct citing patents is overstated by the duplicates."
        )
        return unrepaired_edges(con, received)
    return collapse_duplicate_edges(con, cfg, funnel, received)


def unrepaired_edges(con, received):
    """
    The same columns as the repaired table, without the repair.

    The two branches must agree on the schema. If switching
    citations.deduplicate_edges also changed which columns exist, every reader of
    this table would have to know the setting, and a query written against one
    branch would fail — or worse, resolve a column name it should not have — when
    the other was in force. The flag decides how many rows there are, and
    nothing else about the shape of the answer.

    citation_category_conflicted is false on every row here rather than null:
    without the collapse no pair has been compared with itself, so no conflict
    has been detected. That is a different statement from "no conflict exists",
    and it is why the branch is a diagnostic and not the default.
    """
    received.to_view("citation_edges_unrepaired", replace=True)
    return con.sql("""
        SELECT
            citing_patent_id,
            cited_patent_id,
            citing_grant_date,
            citing_grant_year,
            citing_application_year,
            raw_category                            AS citation_category,
            false                                   AS citation_category_conflicted,
            raw_category = 'cited by examiner'      AS cited_by_examiner
        FROM citation_edges_unrepaired
    """)


def collapse_duplicate_edges(con, cfg, funnel, received):
    """
    One row per distinct (citing, cited) pair, with the disagreements recorded.

    The raw file records 240,051 pairs more than once. 111,875 of those were
    once described as disagreeing with themselves about citation_category, and
    that figure counts "one record has a category, the other has none" as a
    disagreement — which C-64 superseded, because the populated reading is the
    only reading there is. On two different category VALUES it is 209 pairs in
    the raw file, and 121 in this table once the cited side is restricted to the
    universe, which is what citation_category_conflicted counts and what the
    built file reports. Reach is immune to the duplicates themselves,
    because it counts distinct citing patents; knowledge-base share is not,
    because it counts citation ROWS, so every duplicate is a unit of dependence
    that does not exist. That asymmetry is why the defect survived until now —
    the measure that would have exposed it is the one being added.

    The collapse is a GROUP BY rather than a funnel.filter on a row number.
    Both express "keep one row per pair", but a window function partitioned by
    two long text columns over 121 million rows is the shape that swap-thrashed
    this machine for eleven minutes during the Task 2c feasibility work, while a
    hash aggregate spills to disk and does not. The rows it removes are counted
    by the checkpoints on either side of it, which is rule 6's instrument for
    row loss that is not a filter.

    "Spills to disk and does not" holds only while every aggregate in it is one
    DuckDB can spill, which is why the disagreement test is CATEGORIES_DISAGREE
    and not a DISTINCT count — see the comment on that constant. The whole step
    failed on this file five times before that changed.
    """
    funnel.checkpoint("citation edges before duplicates are collapsed", received)
    received.to_view("citation_edges_raw", replace=True)
    collapsed = con.sql("""
        SELECT
            citing_patent_id,
            cited_patent_id,
            -- any_value is safe rather than arbitrary here, and the reason is
            -- measured rather than assumed: all three of these columns are pure
            -- functions of citing_patent_id through the two LEFT JOINs above, and
            -- both source tables hold 0 duplicate patent_id keys
            -- (outputs/default/data_inspection.md). So every row of a
            -- (citing, cited) group carries identical values and there is nothing
            -- to choose between. If a future PatentsView release ever duplicated
            -- a patent_id, this would start picking one of two dates silently —
            -- which is why the assumption is written down here rather than left
            -- as something a reader has to re-derive.
            any_value(citing_grant_date)        AS citing_grant_date,
            any_value(citing_grant_year)        AS citing_grant_year,
            any_value(citing_application_year)  AS citing_application_year,
            -- The category itself, kept only where the records agree. Where they
            -- disagree it is null and the flag beside it says so, so a conflict
            -- can never be mistaken for an unrecorded category (pre-2002) or for
            -- a settled one.
            CASE WHEN {disagree} THEN NULL
                 ELSE max(raw_category) END     AS citation_category,
            {disagree}                          AS citation_category_conflicted,
            {examiner_flag}                     AS cited_by_examiner
        FROM citation_edges_raw
        GROUP BY citing_patent_id, cited_patent_id
    """.format(disagree=CATEGORIES_DISAGREE,
               examiner_flag=EXAMINER_FLAG_RULES[cfg.citations.category_conflict_rule]))
    funnel.note(
        "Duplicate (citing, cited) edges are collapsed to one row each. A duplicate is a "
        "bookkeeping artefact of the source file, not two acts of citing, so counting it twice "
        "would overstate how much of a field's knowledge base rests on the cited work. Where "
        "the duplicated records disagree about who supplied the reference, the examiner flag "
        f"follows citations.category_conflict_rule = {cfg.citations.category_conflict_rule}, "
        "and citation_category_conflicted marks the edge either way."
    )
    funnel.checkpoint("citation edges after duplicates are collapsed", collapsed)
    return collapsed


# ---------------------------------------------------------------------------
# 5. KJL: linking identities, birth years, deaths
# ---------------------------------------------------------------------------

# WHICH ROUTE OWNS A KJL IDENTITY WHEN THE TWO DISAGREE. Decided by Yaren on
# 2026-09-07, on the measurement in `outputs/default/task4b_ladder.csv`: the
# crosswalk route, which reads both identities off one row of PatentsView's own
# release table and assumes nothing about inventor slot order. The coordinate
# route fills in only where the crosswalk has no answer at all, which is the 830
# identities anchored on a Statutory Invention Registration.
#
# The two routes name different people for 27,255 of 1,858,356 identities, 1.5%.
# Measured before the decision was taken: on the occupancy pool at subclass the
# coordinate route gives 3,251 people an in-window death and the crosswalk 3,287,
# and every family x width x arm cell moves the same way, by 0.4% to 1.1%. So the
# coordinate route was the more conservative of the two and the counts it produced
# are a floor. C-23, open since Task 1, is decided here on the treatment side.
#
# THIS BELONGS IN `config/default.yaml` AS `kjl_link.preferred_route` AND IS NOT
# THERE YET, for the same reason MIN_VINTAGE_COLUMN_AGREEMENT below is a constant:
# every analytical key is inside the config hash, `data/interim/` filenames carry
# no hash, and adding one would invalidate every cached dataset in the pipeline —
# including the two citation builds, an hour each. The key goes in at the next
# deliberate rebuild. Unlike that tripwire, this one DOES change an analytical
# result, which is why the paragraph above records what it changes.
PREFERRED_ROUTE = "crosswalk"


def owner_inventor_id_sql():
    """
    The person a KJL identity belongs to, as SQL, under `PREFERRED_ROUTE`.

    Its own function because it is read at three sites — the link table writes
    it, and both the death table and the birth-year table join on it — and a
    rule applied at two of three places is worse than no rule at all: the
    numbers would stay plausible while one table said a death was Ann's and
    another said it was Bea's.
    """
    if PREFERRED_ROUTE == "crosswalk":
        return "COALESCE(x.crosswalk_inventor_id, a.patentsview_inventor_id)"
    if PREFERRED_ROUTE == "coordinate":
        return "COALESCE(a.patentsview_inventor_id, x.crosswalk_inventor_id)"
    raise ValueError(
        f"PREFERRED_ROUTE is '{PREFERRED_ROUTE}', which is neither 'crosswalk' nor "
        f"'coordinate'. It decides which person every death is attached to, so a "
        f"value nobody recognises must stop the run rather than be guessed at.")


def build_kjl_link(con, cfg, funnel):
    """
    Map every KJL inventor identity onto a current PatentsView one, two ways.

    KJL's identifier '3930273-1' names a person by one appearance: patent
    3930273, inventor position 1, counted from one. Current PatentsView counts
    inventor_sequence from zero. Resolving that appearance gives exactly one
    current identity. That is the COORDINATE method, and it is what this
    function computes into patentsview_inventor_id.

    It assumes inventor slot order is the same in both disambiguation releases.
    build_kjl_link_via_crosswalk below computes the same link a second way that
    assumes nothing, by reading both identities off one row of PatentsView's
    own release crosswalk, and its answer is attached here as extra columns.
    Neither is filtered on; which one is authoritative is a Checkpoint 2
    decision. See notes/decisions.md C-23.

    On the release: this repository, and KJL's own paper, described the data as
    built on 20180528. Measured directly it is 20181127 — see C-23. That does
    not affect the coordinate method, which never names a release, but it is
    why config.columns.persistent_kjl_vintage says what it says.

    The two vintages do not agree about who people are, and the disagreement is
    stored rather than resolved, because for prolific inventors — the only
    people who can dominate a field — they agree on the patent set only about a
    quarter of the time. See notes/questions_for_advisor.md, question 18.

    Every patent number arriving from KJL passes through
    normalise_crosswalk_patent_id() first. KJL and PatentsView write Statutory
    Invention Registration numbers differently and the mismatch is silent — see
    src/lib/patent_ids.py for what it costs when it is not repaired.
    """
    columns = cfg.columns
    base = cfg.kjl_link.id_suffix_base

    patent_ids.register(con, repair=cfg.kjl_link.repair_sir_patent_ids)

    identities = con.sql(f"""
        SELECT DISTINCT
            "{columns.kjl_inventor_id}"                                        AS kjl_inventor_id,
            -- The identifier names a person by one appearance, so its patent
            -- half needs the same repair as the patent column itself.
            normalise_crosswalk_patent_id(
                split_part("{columns.kjl_inventor_id}", '-', 1))               AS anchor_patent_id,
            TRY_CAST(split_part("{columns.kjl_inventor_id}", '-', 2) AS INTEGER) - {base}
                                                                               AS anchor_sequence
        FROM read_csv('{kjl(cfg, "kjl_crosswalk")}')
    """)
    identities.to_view("kjl_identities", replace=True)

    patent_ids.check_every_id_repaired(
        con, identities, "anchor_patent_id", label="KJL inventor identifiers",
    )

    con.sql(f"""
        SELECT
            "{columns.patent_id}"                          AS patent_id,
            CAST("{columns.inventor_sequence}" AS INTEGER) AS inventor_sequence,
            "{columns.inventor_id}"                        AS inventor_id
        FROM read_parquet('{raw(cfg, "inventors")}')
    """).to_view("all_inventor_links", replace=True)

    linked = con.sql("""
        SELECT k.kjl_inventor_id, i.inventor_id AS patentsview_inventor_id
        FROM kjl_identities k
        LEFT JOIN all_inventor_links i
               ON i.patent_id = k.anchor_patent_id
              AND i.inventor_sequence = k.anchor_sequence
    """)

    linked = funnel.filter(
        linked, "kjl_identity_resolves",
        "patentsview_inventor_id IS NOT NULL",
        why=(
            "KJL identities that do not resolve to a current PatentsView inventor are dropped. "
            "The identifier names a person by one appearance on one patent, and if that "
            "appearance is no longer in PatentsView — the patent was withdrawn, or the "
            "disambiguation no longer places that person at that position — there is no way "
            "to attach a death to a person we can measure. Note that this is NOT where "
            "Statutory Invention Registrations go: KJL and PatentsView write SIR numbers "
            "differently, and the 830 identities that difference used to cost are repaired "
            "before this filter rather than dropped by it (kjl_link.repair_sir_patent_ids)."
        ),
    )
    linked.to_view("kjl_anchor", replace=True)

    # How far apart the two vintages are for this person, stored rather than acted on.
    comparison = con.sql(f"""
        WITH kjl_patents AS (
            -- Repaired here as well as in `shared` below, and for the same
            -- reason: this count is the denominator that decides whether the
            -- two vintages are called identical. Counting a SIR here that
            -- `shared` could never match is what pushed 2,833 inventors out of
            -- 'identical' before the repair existed.
            SELECT "{columns.kjl_inventor_id}" AS kjl_inventor_id,
                   count(DISTINCT normalise_crosswalk_patent_id(
                       CAST(patent_id AS VARCHAR))) AS kjl_patent_count
            FROM read_csv('{kjl(cfg, "kjl_crosswalk")}') GROUP BY 1
        ),
        pv_patents AS (
            SELECT inventor_id AS patentsview_inventor_id,
                   count(DISTINCT patent_id) AS patentsview_patent_count
            FROM all_inventor_links GROUP BY 1
        ),
        shared AS (
            SELECT a.kjl_inventor_id, count(*) AS shared_patent_count
            FROM kjl_anchor a
            JOIN read_csv('{kjl(cfg, "kjl_crosswalk")}') c
              ON c."{columns.kjl_inventor_id}" = a.kjl_inventor_id
            JOIN all_inventor_links i
              ON i.patent_id = normalise_crosswalk_patent_id(CAST(c.patent_id AS VARCHAR))
             AND i.inventor_id = a.patentsview_inventor_id
            GROUP BY 1
        )
        SELECT
            a.kjl_inventor_id,
            a.patentsview_inventor_id,
            k.kjl_patent_count,
            p.patentsview_patent_count,
            COALESCE(s.shared_patent_count, 0) AS shared_patent_count,
            CASE
                WHEN COALESCE(s.shared_patent_count, 0) = k.kjl_patent_count
                     AND p.patentsview_patent_count = k.kjl_patent_count THEN 'identical'
                WHEN COALESCE(s.shared_patent_count, 0) = k.kjl_patent_count
                     THEN 'patentsview_has_more'
                WHEN COALESCE(s.shared_patent_count, 0) >= 1 THEN 'split_differently'
                ELSE 'no_overlap'
            END AS vintage_agreement
        FROM kjl_anchor a
        JOIN kjl_patents k ON k.kjl_inventor_id = a.kjl_inventor_id
        JOIN pv_patents  p ON p.patentsview_inventor_id = a.patentsview_inventor_id
        LEFT JOIN shared s ON s.kjl_inventor_id = a.kjl_inventor_id
    """)

    if cfg.kjl_link.require_vintage_agreement:
        comparison = funnel.filter(
            comparison, "vintage_agreement_required",
            "vintage_agreement = 'identical'",
            why=(
                "Only inventors whose patent set is identical in both disambiguation vintages "
                "are kept. This is the conservative choice: where the vintages disagree, the "
                "person whose dominance we measure in PatentsView is not exactly the person "
                "whose death KJL records."
            ),
        )
    else:
        funnel.note(
            "Vintage disagreement between KJL and PatentsView is recorded in the column "
            "vintage_agreement and acted on nowhere. Filtering on it later is a config change "
            "(kjl_link.require_vintage_agreement), not a rebuild."
        )

    # The second, independent link. Attached as extra columns rather than
    # replacing patentsview_inventor_id, so that nothing downstream changes
    # behaviour today and the two methods can be compared on the same rows.
    comparison.to_view("anchor_link", replace=True)
    build_kjl_link_via_crosswalk(con, cfg, funnel)

    return con.sql(f"""
        SELECT a.*,
               x.crosswalk_inventor_id,
               x.crosswalk_n_current_ids,
               x.crosswalk_patent_count,
               x.crosswalk_primary_share,
               -- NULL where the crosswalk has no answer, which is the honest
               -- value: 'the two methods disagree' and 'only one method spoke'
               -- are different facts and must not collapse into one flag.
               CASE WHEN x.crosswalk_inventor_id IS NULL THEN NULL
                    ELSE x.crosswalk_inventor_id = a.patentsview_inventor_id
               END AS link_methods_agree,
               -- WHICH PERSON THIS IDENTITY BELONGS TO. See PREFERRED_ROUTE.
               -- Written once, here, so that every later step reads the answer
               -- rather than re-deriving it — two steps deriving it separately
               -- could disagree, and a death would then be attached to two
               -- different people in two tables of one report.
               {owner_inventor_id_sql()} AS owner_inventor_id
        FROM anchor_link a
        LEFT JOIN kjl_crosswalk_link x ON x.kjl_inventor_id = a.kjl_inventor_id
    """)


# The floor below which the identity match is treated as broken rather than
# imperfect. Deliberately a module constant and not a config key, for the same
# reason as TRUNCATION_REFERENCE_YEAR in 00b_inspect_raw.py (notes/decisions.md
# D-2): this is a tripwire on a join, it changes no analytical result, and
# adding a config key would change the config hash and invalidate every cached
# dataset for something that is not an analytical choice.
#
# Measured at 99.76% on 2026-07-30. The floor sits well below that because the
# residual is spelling, not error — see check_identity_match_by_name.
MIN_SURNAME_AGREEMENT = 0.99


def check_identity_match_by_name(con, cfg, funnel):
    """
    Confirm that the person the coordinate lookup returned is the right person.

    Why this check can exist at all. The KJL identity match is a COORDINATE
    lookup, not a name match: '3930273-1' means inventor position 1 on patent
    3930273, and the answer is whoever PatentsView currently places in that
    slot. It never reads a name. That makes the names an INDEPENDENT test of
    the match rather than a restatement of it — KJL's ages file records the
    surname it believes belongs to each identity, so the two can be compared.

    What it would catch. The match assumes inventor slot order is identical in
    both disambiguation vintages. Nobody has verified that, and if a future
    PatentsView release ever reordered inventors on a patent, every affected
    identity would silently resolve to the wrong person — a wrong name, a wrong
    patent set, and a wrong dominance share, with nothing failing anywhere.
    This is the only thing in the pipeline that would notice.

    Why the comparison is on tidied surnames. Raw agreement is 97.7%, and the
    entire shortfall is generational suffixes recorded inconsistently between
    the two sources: KJL has 'mallon, jr.' where PatentsView has 'mallon'.
    Those are the same person. Stripping suffixes and punctuation raises
    agreement to 99.76%, and what remains is accent and spacing loss —
    'mller'/'muller', 'del vecchio'/'delvecchio' — again the same people.
    Comparing raw names would mean setting the floor so low that a real
    reordering could hide underneath it.
    """
    con.execute("""
        CREATE OR REPLACE MACRO tidy_surname(name) AS
            trim(regexp_replace(
                regexp_replace(lower(strip_accents(name)), '[,.]', '', 'g'),
                '\\s+(jr|sr|ii|iii|iv|v)$', '', 'g'))
    """)

    con.sql(f"""
        SELECT DISTINCT inventor_id,
               disambig_inventor_name_last AS patentsview_surname
        FROM read_parquet('{raw(cfg, "inventors")}')
    """).to_view("patentsview_names", replace=True)

    # inventor_last_name is named here rather than in config.columns, matching
    # how the four web-source columns are read in build_birth_years below. The
    # columns block exists because PATENTSVIEW renames columns between releases;
    # KJL is a frozen CC0 Dataverse release (V1) that cannot be renamed under us.
    # Both routes are scored, not only the one PREFERRED_ROUTE uses. The surname
    # test is the one independent check on either — neither route reads a name —
    # so scoring both turns it into evidence about which route is right, and the
    # note below reports the pair. The floor is applied to the route in use.
    agreement = con.sql(f"""
        SELECT
            count(*)                                                       AS compared,
            sum(CASE WHEN tidy_surname(a.inventor_last_name)
                        = tidy_surname(owner.patentsview_surname)
                     THEN 1 ELSE 0 END)                                    AS agreeing,
            count(coordinate.patentsview_surname)                          AS coordinate_compared,
            sum(CASE WHEN tidy_surname(a.inventor_last_name)
                        = tidy_surname(coordinate.patentsview_surname)
                     THEN 1 ELSE 0 END)                                    AS coordinate_agreeing
        FROM read_csv('{kjl(cfg, "kjl_ages")}') a
        JOIN kjl_link_table l ON l.kjl_inventor_id = a."{cfg.columns.kjl_inventor_id}"
        JOIN patentsview_names owner ON owner.inventor_id = l.owner_inventor_id
        LEFT JOIN patentsview_names coordinate
               ON coordinate.inventor_id = l.patentsview_inventor_id
    """).fetchone()

    compared, agreeing, coordinate_compared, coordinate_agreeing = agreement
    if not compared:
        raise ValueError(
            "The identity match could not be checked against names: no KJL ages row joined "
            "to the link table. That is itself a defect — investigate before continuing."
        )

    share = agreeing / compared
    coordinate_share = (coordinate_agreeing / coordinate_compared
                        if coordinate_compared else float("nan"))
    funnel.note(
        f"Identity match checked against recorded names, under the route in use "
        f"('{PREFERRED_ROUTE}'): {agreeing:,} of {compared:,} surnames agree ({share:.2%}). "
        f"The coordinate route alone scores {coordinate_share:.2%} on {coordinate_compared:,} "
        f"rows. Neither route reads a name, so this is an independent test of both, and the "
        f"only thing here that would notice if a PatentsView release reordered inventors on "
        f"a patent — which is exactly the assumption the coordinate route makes and the "
        f"crosswalk route does not."
    )

    if share < MIN_SURNAME_AGREEMENT:
        raise ValueError(
            f"Identity match failed its name check: only {share:.2%} of surnames agree, "
            f"against a floor of {MIN_SURNAME_AGREEMENT:.2%} ({agreeing:,} of {compared:,}).\n\n"
            f"This does not mean the names are untidy. It means the coordinate lookup in "
            f"build_kjl_link is probably returning the WRONG PERSON — most likely because "
            f"this PatentsView vintage orders inventors on a patent differently from the "
            f"20181127 vintage KJL was built on. Every dominance share computed from this "
            f"link would be measured on the wrong people.\n\n"
            f"Do not lower the floor to make this pass. Compare a handful of disagreeing "
            f"identities by hand first — src/steps/00b_inspect_raw.py section 6 is the place "
            f"to look."
        )
    return share


# Floors for the two crosswalk tripwires. Module constants for the same reason
# as MIN_SURNAME_AGREEMENT above: they guard a join, they change no analytical
# result, and putting them in the config would move the config hash.
#
# Measured 2026-07-30: the current-vintage column agrees at exactly 100.0000%
# (24,037,380 of 24,037,380 slots) and KJL identity coverage is 99.947%
# (1,857,526 of 1,858,516; the 990 shortfall is entirely SIR-anchored). The
# coverage floor is set to catch the specific mistake of naming a NEIGHBOURING
# release: 20180528 scores 91.10% and 20190312 91.22%, so 0.99 separates the
# right answer from every wrong one by a wide margin.
MIN_VINTAGE_COLUMN_AGREEMENT = 0.99
MIN_KJL_VINTAGE_COVERAGE = 0.99


def check_current_vintage_column(con, cfg, funnel):
    """
    Confirm that the configured 'current' column really is this PatentsView release.

    The persistent table holds one column per past disambiguation release, and
    the release date is part of the column NAME. Nothing in the file says which
    column corresponds to the inventor table sitting next to it on disk, so if
    PatentsView is re-downloaded and the config is not updated, every crosswalk
    link would silently be made to a previous generation of identities.

    The test is direct: for every inventor slot, the configured column should
    hold the same ID the inventor table holds. Slots whose current ID is empty
    are excluded — the persistent table carries a small number of blank
    duplicate rows for recent patents (19,595 of 24.1M), and they are artefacts
    rather than people. With them excluded the match is exact.
    """
    kjl_col = cfg.columns.persistent_kjl_vintage
    now_col = cfg.columns.persistent_current_vintage

    available = [row[0] for row in con.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{raw(cfg, 'persistent_inventor')}')"
    ).fetchall()]
    for key, col in (("persistent_kjl_vintage", kjl_col),
                     ("persistent_current_vintage", now_col)):
        if col not in available:
            releases = ", ".join(c for c in available if c.startswith("disamb_"))
            raise ValueError(
                f"columns.{key} is set to '{col}', which is not a column of the persistent "
                f"inventor table.\n\nThe releases this file actually contains are:\n"
                f"  {releases}"
            )

    agreement = con.sql(f"""
        SELECT count(*) AS compared,
               sum(CASE WHEN i."{cfg.columns.inventor_id}" = p."{now_col}"
                        THEN 1 ELSE 0 END) AS agreeing
        FROM read_parquet('{raw(cfg, "persistent_inventor")}') p
        JOIN read_parquet('{raw(cfg, "inventors")}') i
          ON i."{cfg.columns.patent_id}" = p."{cfg.columns.patent_id}"
         AND CAST(i."{cfg.columns.inventor_sequence}" AS INTEGER)
           = CAST(p."{cfg.columns.inventor_sequence}" AS INTEGER)
        WHERE nullif(p."{now_col}", '') IS NOT NULL
    """).fetchone()

    compared, agreeing = agreement
    if not compared:
        raise ValueError(
            "The persistent inventor table shares no inventor slot with the inventor table. "
            "One of the two is not the file it is supposed to be — check tables.inventors "
            "and tables.persistent_inventor."
        )

    share = agreeing / compared
    funnel.note(
        f"Persistent crosswalk anchored: column {now_col} matches the inventor table on "
        f"{agreeing:,} of {compared:,} inventor slots ({share:.4%}). This is what proves the "
        f"crosswalk's 'current' column is THIS PatentsView release and not an earlier one."
    )

    if share < MIN_VINTAGE_COLUMN_AGREEMENT:
        raise ValueError(
            f"columns.persistent_current_vintage is set to '{now_col}', but that column "
            f"matches this PatentsView release on only {share:.2%} of inventor slots "
            f"({agreeing:,} of {compared:,}).\n\n"
            f"That column is a DIFFERENT disambiguation release. This usually means "
            f"PatentsView was re-downloaded and the config was not updated: a new release "
            f"adds a new last column to the persistent table.\n\n"
            f"Fix it by finding the column that matches, not by lowering this floor. "
            f"Neighbouring releases score in the eighties and nineties, so a wrong answer "
            f"here still looks plausible."
        )
    return share


def build_kjl_link_via_crosswalk(con, cfg, funnel):
    """
    The second, independent route from a KJL identity to a current PatentsView one.

    Why this exists. build_kjl_link above RECONSTRUCTS the link: it reads
    '3930273-1' as a coordinate, looks up whoever currently sits in inventor
    slot 1 of patent 3930273, and assumes slot order never changed between the
    two disambiguation releases. That assumption is unverifiable from the data
    it uses. This function does not make it. PatentsView publishes a table
    recording, for every inventor slot, the ID that slot held in each past
    release, so both IDs sit on the same row and the link is READ rather than
    inferred. '3930273-1' is not a KJL invention — it is PatentsView's own
    identifier format from that release.

    Which release. Not the one everybody says. KJL's paper and this repository
    both described the data as built on 20180528. Tested against every release
    column in the file, KJL's identities appear in 20181127 at 99.95% —
    1,857,526 of all 1,858,516 — against 91.10% for 20180528. The 990 that
    20181127 misses are, every one of them, SIR-anchored (see below), so across
    the 1,857,526 non-SIR identities the match is 100.0000%. Both figures are
    correct and differ only in whether the SIRs are counted in the denominator.
    See C-23.

    What it cannot do. PatentsView's early disambiguations did not cover
    Statutory Invention Registrations at all — the SIR rows are in the file,
    but their pre-2020 release columns are blank. So the 990 KJL identities
    anchored on a SIR have no crosswalk answer and never will. That is the
    exact mirror of C-21, where those same identities needed a repair to work
    in the coordinate method. Neither method is strictly better everywhere,
    which is the reason both are kept.

    One 2018 identity can map to SEVERAL current inventors, because a later
    disambiguation split the person. The link keeps the primary — the current
    inventor holding most of that person's patents — and stores how many there
    were and what share the primary holds, so the ambiguity is measurable at
    sample selection instead of being silently resolved here.
    """
    kjl_col = cfg.columns.persistent_kjl_vintage
    now_col = cfg.columns.persistent_current_vintage

    # Registered here as well as in build_kjl_link, so this function can be run
    # and tested on its own. Defining the macro twice with the same body costs
    # nothing and is easier to follow than working out where it came from.
    patent_ids.register(con, repair=cfg.kjl_link.repair_sir_patent_ids)

    # KJL writes SIR patent numbers padded with a check character where
    # PatentsView writes them short, so the identifier's patent half needs the
    # same repair as everywhere else (src/lib/patent_ids.py). It buys nothing
    # here — PatentsView never labelled anyone with a SIR in this release — but
    # leaving it out would mean the two methods disagreed about which string a
    # KJL identity even is.
    con.sql(f"""
        SELECT DISTINCT
            "{cfg.columns.kjl_inventor_id}" AS kjl_inventor_id,
            normalise_crosswalk_patent_id(
                split_part("{cfg.columns.kjl_inventor_id}", '-', 1))
                || '-' ||
            split_part("{cfg.columns.kjl_inventor_id}", '-', 2) AS lookup_id
        FROM read_csv('{kjl(cfg, "kjl_crosswalk")}')
    """).to_view("kjl_identities_for_crosswalk", replace=True)

    # Every (past identity, current identity) pair PatentsView itself records,
    # weighted by the number of patents supporting it. A blank release column
    # means the release did not know that slot — a patent granted after it, or
    # a patent type it did not cover — not a person, so blanks are excluded.
    con.sql(f"""
        SELECT "{kjl_col}" AS lookup_id,
               "{now_col}" AS patentsview_inventor_id,
               count(*)    AS pair_patent_count
        FROM read_parquet('{raw(cfg, "persistent_inventor")}')
        WHERE nullif("{kjl_col}", '') IS NOT NULL
          AND nullif("{now_col}", '') IS NOT NULL
        GROUP BY 1, 2
    """).to_view("vintage_pairs", replace=True)

    linked = con.sql("""
        WITH totals AS (
            -- Cast back to a whole number: summing counts widens the type, and
            -- a patent count that prints as 38.0 beside one that prints as 38
            -- reads like the two mean different things. They do not.
            SELECT lookup_id,
                   count(*)                            AS crosswalk_n_current_ids,
                   CAST(sum(pair_patent_count) AS BIGINT) AS crosswalk_patent_count
            FROM vintage_pairs GROUP BY 1
        ),
        ranked AS (
            -- Ties are broken on the identifier itself so that two runs over the
            -- same data always choose the same primary. Without it the answer
            -- would depend on scan order and the link would not be reproducible.
            SELECT lookup_id, patentsview_inventor_id, pair_patent_count,
                   row_number() OVER (
                       PARTITION BY lookup_id
                       ORDER BY pair_patent_count DESC, patentsview_inventor_id ASC
                   ) AS place
            FROM vintage_pairs
        )
        SELECT k.kjl_inventor_id,
               r.patentsview_inventor_id AS crosswalk_inventor_id,
               t.crosswalk_n_current_ids,
               t.crosswalk_patent_count,
               r.pair_patent_count / t.crosswalk_patent_count AS crosswalk_primary_share
        FROM kjl_identities_for_crosswalk k
        JOIN totals t ON t.lookup_id = k.lookup_id
        JOIN ranked r ON r.lookup_id = k.lookup_id AND r.place = 1
    """)
    linked.to_view("kjl_crosswalk_link", replace=True)

    # Coverage is reported rather than filtered: this function feeds a second
    # column onto the link table, and an identity the crosswalk cannot place
    # must stay visible as a null rather than disappear from the sample.
    counts = con.sql("""
        SELECT (SELECT count(DISTINCT kjl_inventor_id) FROM kjl_identities_for_crosswalk),
               (SELECT count(*) FROM kjl_crosswalk_link),
               (SELECT count(*) FROM kjl_crosswalk_link WHERE crosswalk_n_current_ids > 1)
    """).fetchone()
    identities, resolved, split = counts
    share = resolved / identities

    funnel.note(
        f"Second identity link built from PatentsView's own release crosswalk: {resolved:,} "
        f"of {identities:,} KJL identities placed ({share:.3%}), of which {split:,} map to "
        f"more than one current inventor. Unlike the coordinate method this assumes nothing "
        f"about inventor slot order — both identities are read off one row. Stored beside "
        f"the coordinate answer, not substituted for it; which is authoritative is a "
        f"Checkpoint 2 decision."
    )

    if share < MIN_KJL_VINTAGE_COVERAGE:
        raise ValueError(
            f"columns.persistent_kjl_vintage is set to '{kjl_col}', but only {share:.2%} of "
            f"KJL identities appear in that release ({resolved:,} of {identities:,}).\n\n"
            f"KJL's identifiers ARE PatentsView identifiers from one specific release, so "
            f"the right column resolves essentially all of them. The releases either side "
            f"resolve about 91% — high enough to look correct, which is why this check "
            f"exists. Measured on 2026-07-30, the right column is "
            f"disamb_inventor_id_20181127 at 99.95%.\n\n"
            f"Find the column that resolves them; do not lower this floor."
        )
    return share


def resolve_death_records(con, cfg, funnel):
    """
    Turn the death file's candidate matches into records the rest of Task 1 can use.

    The KJL death file is not a list of deaths. It holds 8.46 million candidate
    match records for 535,120 inventors, because the authors searched six
    genealogical sources per location-inventor pair, and most inventors have
    candidates that disagree about the year. Resolving them defines the treated
    sample, so both halves of the rule — the score floor and what to do about a
    remaining tie — are config values. See notes/questions_for_advisor.md,
    question 17.

    Two views come out of this, because the two things that read them need
    different shapes:

      death_records          every candidate that cleared the score floor.
                             Rule B needs all of them: the advisor's rule keeps
                             "the record with the highest death score AMONG
                             THOSE THAT PASS", so a record has to be able to
                             pass on its own before any scores are compared.
      death_by_kjl_identity  one resolved record per KJL identity — the record
                             that dates the death. Rule A takes its birth year,
                             and build_deaths turns it into an event.

    Neither an age filter nor the 1986-2016 death window is applied here. Age is
    Task 4's cutoff, and the window belongs to the event table rather than to the
    birth years: an inventor who died in 2019 is deceased even though the death
    cannot be used as a shock, and Rule A still has to read their record.
    """
    columns = cfg.columns
    deaths_path = kjl(cfg, "kjl_deaths")

    records = con.sql(f"""
        SELECT
            "{columns.kjl_inventor_id}"                        AS kjl_inventor_id,
            "{columns.kjl_death_source}"                       AS source,
            TRY_CAST("{columns.kjl_death_score}" AS DOUBLE)    AS score,
            TRY_CAST("{columns.kjl_death_year}" AS INTEGER)    AS death_year,
            "{columns.kjl_death_date}"                         AS death_date,
            -- KJL writes 0 where the source gave no birth year. Left as 0 it
            -- would read as the year zero, and Rule B would then measure a
            -- 1,950-year disagreement instead of recognising a missing value.
            -- It becomes NULL here, once, at the only place the column is read.
            nullif(TRY_CAST("{columns.kjl_death_birth_year}" AS INTEGER), 0)
                                                               AS record_birth_year
        FROM read_csv('{deaths_path}')
    """)

    records = funnel.filter(
        records, "death_year_present", "death_year > 0",
        why=(
            "Candidate records with no death year are dropped. KJL writes a zero where the "
            "year could not be read from the source, so these carry no information about when "
            "the person died."
        ),
    )

    records = funnel.filter(
        records, "death_match_score_floor", f"score >= {cfg.deaths.min_source_score}",
        why=(
            f"Candidate matches scoring below {cfg.deaths.min_source_score} are dropped. The "
            f"file is a set of candidate matches rather than confirmed deaths, and the score "
            f"is KJL's own measure of how likely a match is correct. The floor matters: at a "
            f"floor of 24 the surviving records agree on a single death year for 97 per cent "
            f"of inventors, at 16 for 72 per cent, and with no floor at all for 47 per cent. "
            f"A false death is worse than a missing one here, because it would put a "
            f"non-event into the treated group. The floor also decides whether Rule B can be "
            f"assessed at all: at 16 every surviving record carries a birth year, at 8 only "
            f"98.1 per cent do, and a record without one fails Rule B by construction."
        ),
    )

    records = funnel.filter(
        records, "death_identity_is_linkable",
        "kjl_inventor_id IN (SELECT kjl_inventor_id FROM kjl_link_table)",
        why=(
            "Death records whose KJL identity does not resolve to a current PatentsView "
            "inventor are dropped. Dominance is measured on PatentsView identities, so a "
            "death that cannot be attached to one cannot become an event, however well "
            "documented it is — and its birth year cannot be attached to a measurable person "
            "either, which is why this happens here rather than inside the event table."
        ),
    )
    records.to_view("death_records", replace=True)

    resolved = con.sql(f"""
        WITH best AS (
            SELECT kjl_inventor_id, max(score) AS best_score
            FROM death_records GROUP BY 1
        ),
        at_best AS (
            SELECT r.kjl_inventor_id, r.death_year, r.death_date, r.record_birth_year,
                   b.best_score
            FROM death_records r
            JOIN best b ON b.kjl_inventor_id = r.kjl_inventor_id
            WHERE r.score = b.best_score
        ),
        counted AS (
            SELECT kjl_inventor_id, death_year, best_score,
                   count(*)        AS records_with_this_year,
                   min(death_date) AS death_date,
                   -- The birth year of the record that DATES the death, so that
                   -- age at death is a subtraction inside one source rather than
                   -- a number assembled from two. Ties go to the earliest year,
                   -- matching min(death_date) just above, so that two runs over
                   -- the same data cannot disagree.
                   list(record_birth_year ORDER BY record_birth_year ASC)
                       FILTER (WHERE record_birth_year IS NOT NULL)[1]
                                   AS record_birth_year,
                   count(*) OVER (PARTITION BY kjl_inventor_id) AS candidate_years
            FROM at_best GROUP BY 1, 2, 3
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY kjl_inventor_id
                                         ORDER BY {TIE_ORDERINGS[cfg.deaths.tie_rule]}) AS pick
            FROM counted
        )
        SELECT kjl_inventor_id, death_year, death_date, record_birth_year, best_score,
               candidate_years, candidate_years > 1 AS death_year_ambiguous
        FROM ranked WHERE pick = 1
    """)

    if cfg.deaths.tie_rule == "drop":
        resolved = funnel.filter(
            resolved, "death_year_unambiguous", "candidate_years = 1",
            why=(
                "Inventors whose best-scoring records still disagree about the year of death "
                "are dropped, because an event study cannot place an event whose date is "
                "uncertain by several years."
            ),
        )
    else:
        funnel.note(
            f"Where the best-scoring records disagreed about the year, the year was chosen by "
            f"the rule '{cfg.deaths.tie_rule}' and the disagreement recorded in the column "
            f"death_year_ambiguous. Nothing is dropped for ambiguity; dropping is a config "
            f"change (deaths.tie_rule), not a rebuild."
        )
    resolved.to_view("death_by_kjl_identity", replace=True)
    return resolved



def build_birth_years(con, cfg, funnel):
    """
    One eligibility gate and two birth-year rules, as Task 1 requires.

    STEP 0 — eligibility, applied to every inventor. KJL's ages file gives each
    inventor a PREFERRED birth year: the single reconciled value produced by the
    heuristics in their Section 4, the `birthyear` column, as distinct from the
    four source-specific years in `radaris_birthyear`, `spokeo_birthyear`,
    `been_birthyear` and `peoplefinders_birthyear` that it was reconciled from.
    Only the preferred year is used. An inventor is retained if that year exists
    and carries a positive match score. A positive score means the inventor was
    matched on at least one geographic or middle-name criterion beyond first and
    last name; zero or below means a name mismatch, no age found, several
    equally-scored matches, or an unspecified error.

    The file scores SOURCES rather than the reconciled year, so the score tested
    is the best of the four — the question the gate asks is whether this person
    was identified at all.

    RULE A, "loose" — assigns a year, and applies to treated and control alike.
    A deceased inventor takes the birth year from the death record that dates
    their death, falling back to the preferred year where the databank supplies
    none. Everyone else takes the preferred year by construction. Nothing is
    required to agree, so eligibility is defined identically for the deceased and
    the living, and this is the only rule that can cover both groups.

    RULE B, "strict" — the deceased only. The death record's birth year must lie
    within birth_years.rule_b_tolerance_years of the preferred year. A death
    record carrying no birth year fails by construction, since agreement cannot
    be assessed. Where several records pass, the one with the highest death score
    is kept.

    Rule B is the advisor's "the two files agree" read literally — the ages file
    against the death file — and the literal reading carries a consequence the
    task list does not state: the test needs a death record, so it has no
    analogue for anyone still alive and cannot define a comparable control group.
    That is why the config has a treated_pool_rule and no general_pool_rule. See
    notes/decisions.md C-25, which withdraws the earlier reading (C-6).

    Both rules are computed and both are stored, so trying the other is a config
    change rather than a rebuild. The gap Rule B tests is stored as a continuous
    number as well, so moving the tolerance at sample
    selection costs nothing.
    """
    ages = kjl(cfg, "kjl_ages")
    rule = cfg.birth_years
    at_least = ">=" if rule.eligibility_score_inclusive else ">"
    tolerance = rule.rule_b_tolerance_years

    # An inventor appears once per city they lived in, so the file is summarised
    # to one row per inventor first. The preferred year is the same on all of an
    # inventor's rows — checked directly: no inventor has two different values —
    # but the scores are not, since 148,800 inventors were matched better in one
    # city than another. So the best row wins the score.
    per_identity = con.sql(f"""
        WITH per_row AS (
            SELECT
                inventor_id                    AS kjl_inventor_id,
                TRY_CAST(birthyear AS INTEGER) AS preferred_birth_year,
                -- greatest() ignores nulls in DuckDB and returns null only when
                -- every argument is null, which is exactly the meaning wanted
                -- here: a source that was never searched must not win the
                -- maximum, but an inventor no source found must score nothing.
                greatest(TRY_CAST(radaris_score       AS DOUBLE),
                         TRY_CAST(spokeo_score        AS DOUBLE),
                         TRY_CAST(been_score          AS DOUBLE),
                         TRY_CAST(peoplefinders_score AS DOUBLE))
                                               AS best_source_score,
                -- Two diagnostics, deliberately not part of any rule. They
                -- record how much the reconciled year rests on: how many of the
                -- four sources produced a year at all, and how many produced
                -- exactly the year KJL settled on.
                (CASE WHEN TRY_CAST(radaris_birthyear       AS INTEGER) IS NOT NULL
                      THEN 1 ELSE 0 END
               + CASE WHEN TRY_CAST(spokeo_birthyear        AS INTEGER) IS NOT NULL
                      THEN 1 ELSE 0 END
               + CASE WHEN TRY_CAST(been_birthyear          AS INTEGER) IS NOT NULL
                      THEN 1 ELSE 0 END
               + CASE WHEN TRY_CAST(peoplefinders_birthyear AS INTEGER) IS NOT NULL
                      THEN 1 ELSE 0 END)       AS sources_with_a_year,
                (CASE WHEN TRY_CAST(radaris_birthyear       AS INTEGER)
                         = TRY_CAST(birthyear AS INTEGER) THEN 1 ELSE 0 END
               + CASE WHEN TRY_CAST(spokeo_birthyear        AS INTEGER)
                         = TRY_CAST(birthyear AS INTEGER) THEN 1 ELSE 0 END
               + CASE WHEN TRY_CAST(been_birthyear          AS INTEGER)
                         = TRY_CAST(birthyear AS INTEGER) THEN 1 ELSE 0 END
               + CASE WHEN TRY_CAST(peoplefinders_birthyear AS INTEGER)
                         = TRY_CAST(birthyear AS INTEGER) THEN 1 ELSE 0 END)
                                               AS sources_backing_preferred
            FROM read_csv('{ages}')
        )
        SELECT
            kjl_inventor_id,
            max(preferred_birth_year)      AS preferred_birth_year,
            max(best_source_score)         AS best_source_score,
            max(sources_with_a_year)       AS sources_with_a_year,
            max(sources_backing_preferred) AS sources_backing_preferred
        FROM per_row GROUP BY 1
    """)

    # Step 0 is two filters rather than one because they answer different
    # questions, and the funnel is where that difference gets recorded: "no age
    # was found for this person" and "an age was found but the person behind it
    # was never confidently identified" are not the same missing data.
    eligible = funnel.filter(
        per_identity, "step0_has_preferred_birth_year",
        "preferred_birth_year IS NOT NULL",
        why=(
            "Inventors with no preferred birth year are dropped. The preferred year is KJL's "
            "single reconciled value, computed by the heuristics in their Section 4, and it is "
            "the only birth year this design uses from the ages file — not the four "
            "source-specific years it was reconciled from. An inventor without one cannot be "
            "classified as dying prematurely or not, on either rule."
        ),
    )

    eligible = funnel.filter(
        eligible, "step0_match_score_positive",
        f"best_source_score {at_least} {rule.eligibility_min_score}",
        why=(
            f"Inventors whose best web-source match score is not "
            f"{'at least' if rule.eligibility_score_inclusive else 'above'} "
            f"{rule.eligibility_min_score} are dropped. A positive score means the inventor was "
            f"matched on at least one geographic or middle-name criterion beyond first and last "
            f"name; zero or negative values mark a name mismatch, no age found, several equally "
            f"scored matches, or an unspecified error. The ages file scores sources rather than "
            f"the reconciled year, so the score tested is the best of the four, and the best of "
            f"an inventor's location rows: if any match identified the person, the person was "
            f"identified. This gate applies to the deceased and the living alike, which is what "
            f"lets Rule A define a control group on the same terms as the treated."
        ),
    )

    eligible = funnel.filter(
        eligible, "birth_year_identity_is_linkable",
        "kjl_inventor_id IN (SELECT kjl_inventor_id FROM kjl_link_table)",
        why=(
            "Birth years whose KJL identity does not resolve to a current PatentsView "
            "inventor are dropped, for the same reason as the deaths: an age that cannot be "
            "attached to a measurable person cannot be used."
        ),
    )
    eligible.to_view("eligible_birth_years", replace=True)

    # How far the death file is from the ages file for this person, over EVERY
    # candidate record and before any tolerance is applied. Stored continuous so
    # that Rule B's tolerance can be moved at sample selection without
    # rebuilding: min_gap_years <= t reproduces Rule B at any tolerance t.
    con.sql("""
        SELECT
            r.kjl_inventor_id,
            min(abs(r.record_birth_year - e.preferred_birth_year)) AS min_gap_years,
            count(*)                                               AS records_with_a_birth_year
        FROM death_records r
        JOIN eligible_birth_years e ON e.kjl_inventor_id = r.kjl_inventor_id
        WHERE r.record_birth_year IS NOT NULL
        GROUP BY 1
    """).to_view("rule_b_gaps", replace=True)

    # Rule B itself: the records that pass, and the highest-scoring one of them.
    # The ordering is written out rather than left to arg_max because arg_max
    # breaks ties arbitrarily and DuckDB evaluates in parallel, which would make
    # the same config produce different answers on different runs.
    con.sql(f"""
        SELECT
            r.kjl_inventor_id,
            count(*)                                                       AS records_passing,
            max(r.score)                                                   AS death_score_strict,
            list(r.record_birth_year ORDER BY r.score DESC, r.death_year ASC,
                                              r.record_birth_year ASC)[1]  AS birth_year_strict,
            list(r.death_year        ORDER BY r.score DESC, r.death_year ASC,
                                              r.record_birth_year ASC)[1]  AS death_year_strict
        FROM death_records r
        JOIN eligible_birth_years e ON e.kjl_inventor_id = r.kjl_inventor_id
        WHERE r.record_birth_year IS NOT NULL
          AND abs(r.record_birth_year - e.preferred_birth_year) <= {tolerance}
        GROUP BY 1
    """).to_view("rule_b_records", replace=True)

    with_rules = con.sql("""
        SELECT
            e.kjl_inventor_id,
            e.preferred_birth_year,
            e.best_source_score,
            e.sources_with_a_year,
            e.sources_backing_preferred,
            d.kjl_inventor_id IS NOT NULL              AS is_deceased,
            d.record_birth_year                        AS death_record_birth_year,
            -- Carried so that the collapse onto one PatentsView person below can
            -- choose the same death record the event table will choose.
            d.best_score                               AS death_match_score,
            d.death_year                               AS death_record_year,
            -- RULE A. The death record's year for the deceased; the preferred
            -- year for everyone else, and as the fallback for a deceased
            -- inventor whose databank supplied no birth year.
            COALESCE(d.record_birth_year, e.preferred_birth_year) AS birth_year_loose,
            CASE WHEN d.record_birth_year IS NOT NULL THEN 'death_record'
                 ELSE 'age_file' END                   AS birth_year_loose_source,
            -- RULE B. Null for the living, and null is the honest value: "the
            -- test does not apply to this person" and "this person failed the
            -- test" are different facts and must not collapse into one flag.
            CASE WHEN d.kjl_inventor_id IS NULL THEN NULL
                 ELSE b.kjl_inventor_id IS NOT NULL END AS passes_rule_b,
            b.birth_year_strict,
            b.death_year_strict,
            b.records_passing                          AS records_passing_rule_b,
            g.min_gap_years,
            g.records_with_a_birth_year
        FROM eligible_birth_years e
        LEFT JOIN death_by_kjl_identity d ON d.kjl_inventor_id = e.kjl_inventor_id
        LEFT JOIN rule_b_records       b ON b.kjl_inventor_id = e.kjl_inventor_id
        LEFT JOIN rule_b_gaps          g ON g.kjl_inventor_id = e.kjl_inventor_id
    """)
    with_rules.to_view("birth_years_by_kjl_id", replace=True)

    deceased, passing, from_record = con.sql("""
        SELECT count(*) FILTER (WHERE is_deceased),
               count(*) FILTER (WHERE passes_rule_b),
               count(*) FILTER (WHERE birth_year_loose_source = 'death_record')
        FROM birth_years_by_kjl_id
    """).fetchone()
    # A pool with nobody in it is normal in sample mode and in the tests, so the
    # rate is only quoted when there is something to divide by.
    rate = f" ({passing / deceased:.1%} of them)" if deceased else ""
    funnel.note(
        f"Of the KJL identities that cleared Step 0, {deceased:,} have a death record at or "
        f"above the score floor. Rule A took the birth year from that record for "
        f"{from_record:,} of them and fell back to the ages file for the rest; every identity "
        f"without a death record keeps the preferred year by construction. Rule B — the death "
        f"record's birth year within {tolerance} years of the preferred year — is passed by "
        f"{passing:,} of the deceased{rate}, and is null rather than false for everyone else, "
        f"because a living inventor does not fail a test that cannot be applied to them."
    )

    # One current PatentsView identity can inherit several KJL identities, so the
    # whole row is taken from ONE of them rather than each column being maximised
    # separately. A row assembled column by column could carry a birth year from
    # one identity and a Rule B verdict from another, and describe nobody.
    #
    # WHICH identity wins is not a tidying detail, it decides whether Rule A
    # applies at all. A person can hold one identity with a death record and
    # another without, and picking on patent count alone picked the one without
    # for 16,278 people who had in fact died — so Rule A quietly handed them the
    # ages-file year when the rule says to use their death record. Deceased
    # identities therefore sort first, and among them the best-scoring death
    # record, which is the same record the event table will choose. Patent count
    # only decides between identities that are alike on all of that, which for
    # the living is every comparison.
    per_person = con.sql("""
        WITH joined AS (
            -- owner_inventor_id, not patentsview_inventor_id, and for the same
            -- reason as in build_deaths: the birth year has to follow the same
            -- route rule the death does, or a person could be handed a year
            -- derived from an identity the rule says is somebody else's.
            SELECT a.owner_inventor_id AS inventor_id, a.kjl_patent_count, b.*
            FROM kjl_link_table a
            JOIN birth_years_by_kjl_id b ON b.kjl_inventor_id = a.kjl_inventor_id
        ),
        ranked AS (
            -- Ties go to the lower identifier so that the answer does not depend
            -- on the order DuckDB happens to scan in.
            SELECT *, row_number() OVER (PARTITION BY inventor_id
                                         ORDER BY is_deceased DESC,
                                                  death_match_score DESC NULLS LAST,
                                                  death_record_year ASC NULLS LAST,
                                                  kjl_patent_count DESC,
                                                  kjl_inventor_id ASC) AS pick
            FROM joined
        ),
        conflicts AS (
            SELECT inventor_id,
                   count(DISTINCT kjl_inventor_id)      AS kjl_identities,
                   count(DISTINCT birth_year_loose) > 1 AS birth_year_conflict
            FROM joined GROUP BY 1
        )
        SELECT r.* EXCLUDE (pick, kjl_patent_count),
               c.kjl_identities,
               c.birth_year_conflict
        FROM ranked r
        JOIN conflicts c ON c.inventor_id = r.inventor_id
        WHERE r.pick = 1
    """)
    # Rows fall here without anything being filtered: several KJL identities
    # collapse onto one current person. The checkpoint records the collapse so
    # the drop in the funnel is not mistaken for a filter.
    funnel.checkpoint("birth years, one row per PatentsView inventor", per_person)
    return per_person


def build_deaths(con, cfg, funnel):
    """
    One usable death per person, inside the window, with age attached.

    The candidate records were already resolved into one record per KJL identity
    by resolve_death_records. What is left is the part that makes a death an
    EVENT: it has to fall inside the usable window, and one person has to be one
    event, because inference clusters at the death.

    No age filter is applied here either. Age is computed and stored; premature
    death is Task 4's cutoff, not Task 1's, and applying it now would discard the
    events that make the robustness check at 65 possible.
    """
    linked = con.sql("""
        SELECT
            -- The person this identity belongs to under PREFERRED_ROUTE, written
            -- by build_kjl_link. NOT patentsview_inventor_id: that is the
            -- coordinate route's answer alone, and where the two routes disagree
            -- it hands the death to a different person.
            a.owner_inventor_id AS inventor_id,
            a.kjl_inventor_id,
            a.vintage_agreement,
            a.kjl_patent_count,
            a.patentsview_patent_count,
            -- Carried through so the two identity methods can be compared on
            -- EVENTS rather than on all 1.9M identities. The number that
            -- matters at Checkpoint 2 is how many usable deaths would change
            -- person, not how many inventors would.
            a.crosswalk_inventor_id,
            a.link_methods_agree,
            d.death_year, d.death_date, d.record_birth_year, d.best_score,
            d.candidate_years, d.death_year_ambiguous
        FROM death_by_kjl_identity d
        JOIN kjl_link_table a ON a.kjl_inventor_id = d.kjl_inventor_id
    """)

    linked = funnel.filter(
        linked, "death_year_in_window",
        f"death_year BETWEEN {cfg.deaths.year_min} AND {cfg.deaths.year_max}",
        why=(
            f"Only deaths between {cfg.deaths.year_min} and {cfg.deaths.year_max} are usable. "
            f"The advisor's reason: we need {cfg.windows.pre_years} years of patent data "
            f"before each death — and the data starts in {cfg.patent_universe.first_year} "
            f"plus a {cfg.windows.dominance_window_years}-year measurement window — and "
            f"{cfg.windows.post_years} reasonably complete years after, since filings are "
            f"only complete through about {cfg.patent_universe.filings_complete_through}. "
            f"Note that this filter builds the EVENT table only: the birth-year table was "
            f"built before it, so an inventor who died outside the window is still recorded "
            f"as deceased and still gets their death record's birth year under Rule A."
        ),
    )
    linked.to_view("deaths_in_window", replace=True)

    # One current PatentsView person can inherit several KJL identities, because
    # the 2018 disambiguation split people the current one merges. Each of those
    # identities brings its own death record, so without this step one person
    # would enter the design as several events — and inference clusters at the
    # death precisely because one person is one event.
    ranked_identities = con.sql("""
        SELECT *,
               count(*)      OVER (PARTITION BY inventor_id) AS kjl_identities,
               count(DISTINCT death_year)
                             OVER (PARTITION BY inventor_id) > 1 AS death_year_conflict,
               row_number()  OVER (PARTITION BY inventor_id
                                   ORDER BY best_score DESC, death_year ASC,
                                            kjl_inventor_id ASC) AS pick
        FROM deaths_in_window
    """)
    one_per_person = funnel.filter(
        ranked_identities, "one_death_per_person", "pick = 1",
        why=(
            "Where several KJL identities resolve to the same current PatentsView inventor, "
            "only the best-scoring death record is kept, so that one person is one event. "
            "The older disambiguation split people whom the current one merges, and leaving "
            "the duplicates in would enter the same death into the design more than once — "
            "which is exactly what clustering at the death is meant to prevent. How many "
            "identities were merged, and whether they disagreed about the year, are kept in "
            "the columns kjl_identities and death_year_conflict."
        ),
    )
    one_per_person.to_view("one_death_per_person", replace=True)

    # Age is computed and stored, never filtered on here. Which rule supplies it
    # follows birth_years.treated_pool_rule, because these are the potential
    # treated inventors; Task 4a asks for the looser list.
    #
    # Rule A is applied to the EVENT rather than copied from the general-pool
    # birth-year table. The two tables are both keyed on the person but are built
    # from different sets of KJL identities — a person can hold an identity that
    # cleared Step 0 and a different one that supplied this death — so the
    # general-pool answer is not always about the record that dates this event.
    # Age at death has to be a subtraction inside ONE death record, or it is a
    # number belonging to nobody, so the event computes its own and the
    # general-pool answer is carried alongside for comparison.
    treated_list = f"birth_year_{cfg.birth_years.treated_pool_rule}"
    events = con.sql(f"""
        WITH with_rule_a AS (
            SELECT
                d.inventor_id,
                d.kjl_inventor_id,
                d.kjl_identities,
                d.death_year,
                d.death_date,
                d.best_score            AS death_match_score,
                d.candidate_years       AS death_candidate_years,
                d.death_year_ambiguous,
                d.death_year_conflict,
                d.vintage_agreement,
                d.kjl_patent_count,
                d.patentsview_patent_count,
                d.crosswalk_inventor_id,
                d.link_methods_agree,
                d.record_birth_year     AS event_record_birth_year,
                b.preferred_birth_year,
                b.birth_year_loose      AS birth_year_loose_general_pool,
                b.birth_year_strict,
                b.passes_rule_b,
                b.min_gap_years,
                b.sources_backing_preferred,
                -- Rule A for this event: the birth year on the record that dates
                -- the death, falling back to the reconciled year where the
                -- databank supplied none. Step 0 is still respected, because an
                -- inventor who failed it has no row in the birth-year table and
                -- so has no reconciled year to fall back to and no age at all.
                CASE WHEN b.inventor_id IS NULL THEN NULL
                     ELSE COALESCE(d.record_birth_year, b.preferred_birth_year)
                END                     AS birth_year_loose,
                CASE WHEN b.inventor_id IS NULL THEN NULL
                     WHEN d.record_birth_year IS NOT NULL THEN 'death_record'
                     ELSE 'age_file'
                END                     AS birth_year_loose_source
            FROM one_death_per_person d
            LEFT JOIN birth_years_table b ON b.inventor_id = d.inventor_id
        )
        SELECT *,
               death_year - birth_year_loose  AS age_at_death_loose,
               death_year - birth_year_strict AS age_at_death_strict,
               death_year - {treated_list}    AS age_at_death,
               -- Not a threshold, a logical impossibility: the birth year and
               -- the death record cannot belong to the same person. Flagged,
               -- not dropped, because which ages are usable is Task 4's decision.
               (death_year - {treated_list}) < 0 AS age_impossible,
               -- Whether the general-pool table agrees with this event about the
               -- Rule A year. Nothing downstream depends on it, since the age
               -- above is computed from the event either way, but a large number
               -- here would mean the two tables describe different people.
               birth_year_loose_general_pool IS NOT DISTINCT FROM birth_year_loose
                                              AS general_pool_year_agrees
        FROM with_rule_a
    """)

    events.to_view("events_for_note", replace=True)
    differing, comparable = con.sql("""
        SELECT count(*) FILTER (WHERE NOT general_pool_year_agrees),
               count(*) FILTER (WHERE birth_year_loose IS NOT NULL)
        FROM events_for_note
    """).fetchone()
    if differing:
        funnel.note(
            f"On {differing:,} of {comparable:,} events the general-pool birth-year table "
            f"gives a different Rule A year from the one computed here, because the person "
            f"carries several KJL identities and the identity that table answers for is not "
            f"the one that supplied this death. The age at death above is computed from THIS "
            f"event's own death record, so it is a subtraction inside one source in every "
            f"case; the other answer is kept in birth_year_loose_general_pool and the "
            f"disagreement flagged in general_pool_year_agrees."
        )
    return events


# ---------------------------------------------------------------------------
# The note the advisor asked for — generated, never typed
# ---------------------------------------------------------------------------

def _birth_year_section(con, cfg):
    """
    Step 0, Rule A and Rule B on this run's own numbers.

    Generated, never typed. Three things decide whether the birth-year rules are
    usable, and none of them can be known in advance: how many people the
    eligibility gate keeps, how often Rule A actually reaches the death file
    rather than falling back to the ages file, and how many of the deceased
    survive Rule B. That last number is the size of the strict robustness check —
    if it collapses, the check is not available and Checkpoint 2 needs to know.

    The gap table is the evidence for the tolerance. It is printed at every
    width, not just at the configured one, because the argument for ±2 is that
    the modal gap is 1 rather than 0, and that argument is only visible if the
    distribution is shown rather than summarised.
    """
    births = cfg.birth_years
    people, deceased, from_record, passing, conflicts = con.sql("""
        SELECT count(*),
               count(*) FILTER (WHERE is_deceased),
               count(*) FILTER (WHERE birth_year_loose_source = 'death_record'),
               count(*) FILTER (WHERE passes_rule_b),
               count(*) FILTER (WHERE birth_year_conflict)
        FROM birth_years_table
    """).fetchone()

    lines = [
        "## Birth years: one gate, two rules",
        "",
        f"**Step 0.** An inventor is eligible if KJL's reconciled `birthyear` exists and the "
        f"best of the four web-source scores is "
        f"{'at least' if births.eligibility_score_inclusive else 'above'} "
        f"{births.eligibility_min_score}. The reconciled value is the one produced by the "
        f"heuristics in their Section 4 — not the source-specific years from Radaris, Spokeo, "
        f"BeenVerified or PeopleFinders. A positive score means the inventor was matched on at "
        f"least one geographic or middle-name criterion beyond first and last name; zero or "
        f"below marks a name mismatch, no age found, several equally-scored matches, or an "
        f"unspecified error.",
        "",
        f"- Eligible inventors with a current PatentsView identity: **{people:,}**",
        f"- Of those, with a death record at or above the score floor: **{deceased:,}**",
        f"- Rule A took the birth year from the death record for **{from_record:,}** of them"
        + (f", and fell back to the ages file for the remaining {deceased - from_record:,} whose "
           f"databank supplied none" if from_record < deceased
           else " — every one, so the fallback to the ages file never fired"),
        f"- Inventors whose several KJL identities disagree about the Rule A year: "
        f"**{conflicts:,}** (flagged, not resolved)",
        "",
    ]

    if not deceased:
        lines += [
            "No eligible inventor has a death record in this run, so Rule B could not be "
            "assessed. In a full run that would be a defect; in sample mode it only means the "
            "CPC section is small.",
            "",
        ]
        return lines

    lines += [
        f"**Rule B** — the death record's birth year within ±{births.rule_b_tolerance_years} "
        f"years of the reconciled year — is passed by **{passing:,}** of the {deceased:,} "
        f"deceased ({passing / deceased:.1%}). It is null rather than false for everyone "
        f"still alive: a test that needs a death record cannot be failed by someone who has "
        f"no death record, and that is precisely why Rule B cannot define a control group.",
        "",
        "How far apart the two files are, over every candidate record, before any tolerance "
        "is applied:",
        "",
        "| smallest gap (years) | deceased inventors | share |",
        "| --- | --- | --- |",
    ]

    # Top-coded so that a long thin tail cannot push the first rows off the page —
    # and the first rows are the whole argument. The config validator has
    # already refused the run if the top code would swallow the row that decides
    # Rule B, so the table below can be read as evidence for the tolerance.
    rows = con.sql(f"""
        SELECT least(min_gap_years, {births.gap_table_top_code}) AS gap,
               count(*) AS inventors
        FROM birth_years_table
        WHERE is_deceased AND min_gap_years IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    for gap, n in rows:
        # The label is derived from the key rather than typed, so that it cannot
        # go on saying "6 or more" over a table top-coded somewhere else.
        label = (f"{births.gap_table_top_code} or more"
                 if gap >= births.gap_table_top_code else str(gap))
        lines.append(f"| {label} | {n:,} | {n / deceased:.1%} |")

    lines += [
        "",
        "Read the first two rows together. A gap of exactly one year is normally more common "
        "than exact agreement, and the reason is mechanical rather than random: the web "
        "directories report an *age*, so a birth year derived from one is short by a year for "
        "anybody whose birthday had not yet passed when the page was scraped. That is the "
        f"argument for a tolerance of {births.rule_b_tolerance_years} rather than 0 — an "
        f"exact-agreement rule would discard most genuine matches as disagreements. The gap "
        f"is stored per inventor as `min_gap_years`, so a different tolerance is a filter at "
        f"sample selection, not a rebuild.",
        "",
    ]
    return lines


def _link_method_section(con, cfg):
    """
    The two identity methods side by side — the evidence Checkpoint 2 needs.

    Generated, never typed. The question this answers is not "do the methods
    agree" in the abstract but "how many people would change identity, and are
    they the people the coordinate method was already unsure about". If
    disagreement sits inside vintage_agreement = 'identical', something is
    wrong with one of the two methods and the numbers below will say so.
    """
    totals = con.sql("""
        SELECT count(*)                                                        AS identities,
               count(crosswalk_inventor_id)                                    AS placed,
               sum(CASE WHEN link_methods_agree THEN 1 ELSE 0 END)             AS agree,
               sum(CASE WHEN crosswalk_n_current_ids > 1 THEN 1 ELSE 0 END)    AS split
        FROM kjl_link_table
    """).fetchone()
    identities, placed, agree, split = totals

    kjl_release = cfg.columns.persistent_kjl_vintage.replace("disamb_inventor_id_", "")
    now_release = cfg.columns.persistent_current_vintage.replace("disamb_inventor_id_", "")

    lines = [
        "## Two ways of linking KJL to PatentsView",
        "",
        f"**Which PatentsView release KJL is built on: `{kjl_release}`.** This is measured, and it "
        f"corrects what KJL's own paper says (20180528) and what this repository said until "
        f"2026-07-30. Tested against all 28 release columns, KJL's identifiers appear in "
        f"`{kjl_release}` at 99.95% and in 20180528 at 91.10%; every neighbouring release sits "
        f"near 90%, which is high enough that a wrong answer reads as ordinary cross-vintage "
        f"attrition rather than as an error. The full table of all 28 is in "
        f"`data_inspection.md`, section 6, where it is re-derived from the data on every run. "
        f"The current release is `{now_release}`, confirmed against the inventor table before "
        f"anything below was built. See `notes/decisions.md` C-23.",
        "",
        f"The coordinate method resolves a KJL identity by looking up who now sits in the "
        f"inventor slot its identifier names. The crosswalk method reads both identities off "
        f"one row of PatentsView's own release table and assumes nothing about slot order. "
        f"Both are stored; neither is filtered on.",
        "",
        f"- Identities linked by the coordinate method: **{identities:,}**",
        f"- Also placed by the crosswalk: **{placed:,}** ({placed / identities:.2%})",
        f"- The two methods name the same person: **{agree:,}** "
        f"({agree / placed:.2%} of those placed)",
        f"- 2018 identities the current release splits across several people: **{split:,}**",
        "",
        "| vintage agreement | identities | crosswalk placed | methods agree | agree % |",
        "| --- | --- | --- | --- | --- |",
    ]

    rows = con.sql("""
        SELECT vintage_agreement,
               count(*)                                             AS identities,
               count(crosswalk_inventor_id)                         AS placed,
               sum(CASE WHEN link_methods_agree THEN 1 ELSE 0 END)  AS agree
        FROM kjl_link_table GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    for label, n, n_placed, n_agree in rows:
        pct = f"{n_agree / n_placed:.2%}" if n_placed else "—"
        lines.append(f"| {label} | {n:,} | {n_placed:,} | {n_agree:,} | {pct} |")

    lines += [
        "",
        "Read the last column, not the first. Disagreement concentrated in "
        "`split_differently` and `no_overlap` is the expected result: those are exactly the "
        "identities the coordinate method had no way to get right, because it answers with "
        "whoever holds one patent. Disagreement inside `identical` would instead mean one of "
        "the two methods is broken.",
        "",
    ]
    return lines


def write_note(cfg, funnel, produced, con):
    """
    One page: the files used and the rules fixed.

    Generated from the config and the funnel log rather than written by hand, so
    it cannot drift away from what the code actually did.
    """
    universe = cfg.patent_universe
    lines = [
        "# Task 1 — the files used and the rules fixed",
        "",
        f"Config `{cfg.name}`, hash `{cfg.hash}`. Generated by `src/02_build_patent_tables.py` from "
        f"the configuration and the run's own funnel log, so it describes what the code did "
        f"rather than what anyone intended it to do.",
        "",
    ]
    if cfg.sample_mode:
        lines += [
            f"> **Sample mode is ON.** Everything below is restricted to CPC section "
            f"`{cfg.sample_cpc_section}`. These are debugging numbers, not results. Set "
            f"`sample_mode: false` in `config/default.yaml` for the full run.",
            "",
        ]

    lines += ["## Files used", "", "| file | role | rows |", "| --- | --- | --- |"]
    for name, role, path in produced:
        lines.append(f"| `{Path(path).name}` | {role} | {count_rows(con, path):,} |")
    lines += [
        "",
        "Sources: PatentsView bulk tables (converted to Parquet by step 01) and the "
        "Kaltenberg–Jaffe–Lachman ages, deaths and crosswalk files, read directly and never "
        "modified. Provenance for both is in `docs/data_sources.md`.",
        "",
    ]

    # Four of the rules below can be switched off by a flag, so their text is
    # built from the flag rather than assumed. A note that states a rule the run
    # did not actually apply is worse than no note, because it reads as evidence.
    if universe.drop_incomplete_filing_years:
        years_kept = f"**{universe.first_year}–{universe.filings_complete_through}**"
    else:
        years_kept = (f"**{universe.first_year} onwards** — recent filing years are NOT "
                      f"truncated, so the last years are incomplete")
    withdrawn_rule = "**dropped**" if universe.drop_withdrawn else "**kept**"
    births = cfg.birth_years
    step0_rule = (f"best of the four web-source scores "
                  f"{'≥' if births.eligibility_score_inclusive else '>'} "
                  f"**{births.eligibility_min_score}**, on KJL's reconciled `birthyear`")
    if cfg.kjl_link.require_vintage_agreement:
        vintage_rule = "**required** — inventors whose patent sets disagree are dropped"
    else:
        vintage_rule = "**not required** — disagreement is stored per inventor, not filtered on"

    lines += [
        "## Rules fixed",
        "",
        "Every rule below is a value in `config/default.yaml`. Changing one is a YAML edit "
        "and a re-run; none of them is written into the code.",
        "",
        "| rule | setting | value |",
        "| --- | --- | --- |",
        f"| Patents dated by | `patent_universe.date_by` | **{universe.date_by}** |",
        f"| Counting | `counting.method` | **{cfg.counting.method}** |",
        f"| CPC assignment | `cpc.assignment` | **{cfg.cpc.assignment}** "
        f"(sequence {cfg.cpc.primary_sequence}) |",
        f"| CPC vintage | `cpc.table` | **{cfg.cpc.table}** |",
        f"| Field definition | `fields.strategy` | **{cfg.fields.strategy}** |",
        f"| Patent types | `patent_universe.patent_types` | **{', '.join(universe.patent_types)}** |",
        f"| Withdrawn patents | `patent_universe.drop_withdrawn` | {withdrawn_rule} |",
        f"| Years kept | `patent_universe.first_year`, `.filings_complete_through`, "
        f"`.drop_incomplete_filing_years` | {years_kept} |",
        f"| Deaths kept | `deaths.year_min`, `.year_max` | "
        f"**{cfg.deaths.year_min}–{cfg.deaths.year_max}** |",
        f"| Death match floor | `deaths.min_source_score` | **{cfg.deaths.min_source_score}** |",
        f"| Death year ties | `deaths.tie_rule` | **{cfg.deaths.tie_rule}** |",
        f"| Birth-year eligibility (Step 0) | `birth_years.eligibility_min_score`, "
        f"`.eligibility_score_inclusive` | {step0_rule} |",
        f"| Rule B tolerance | `birth_years.rule_b_tolerance_years` | "
        f"**± {births.rule_b_tolerance_years} years** between the death record and the "
        f"ages file |",
        f"| Treated pool uses | `birth_years.treated_pool_rule` | "
        f"**{births.treated_pool_rule}** (Rule "
        f"{'B, strict' if births.treated_pool_rule == 'strict' else 'A, loose'}) |",
        f"| KJL identity link | `kjl_link.id_suffix_base` | "
        f"**one-based against PatentsView's zero-based sequence** |",
        f"| KJL vintage disagreement | `kjl_link.require_vintage_agreement` | {vintage_rule} |",
        "",
        "### What is deliberately not fixed here",
        "",
        "- **No dominance.** No share, rank or margin is computed. That is Task 2.",
        "- **No age cutoff.** Age at death is computed and stored as a number; premature "
        "death is Task 4's cutoff, and applying it here would discard the events that make "
        "the robustness check at 65 possible.",
        "- **No suddenness filter.** KJL carries no cause of death, so one cannot be built "
        "from these sources at all (see `notes/decisions.md`, B-3).",
        "",
    ]

    lines += _birth_year_section(con, cfg)
    lines += _link_method_section(con, cfg)

    lines += ["## What the funnel did", "", "| step | rows before | rows after | dropped |",
              "| --- | --- | --- | --- |"]
    for entry in funnel.steps:
        if entry["kind"] == "note" or entry["rows_before"] is None:
            continue
        dropped = entry["rows_before"] - entry["rows_after"]
        share = f"{dropped / entry['rows_before']:.1%}" if entry["rows_before"] else "—"
        label = entry["name"] if entry["kind"] == "filter" else f"_{entry['name']}_"
        lines.append(f"| {label} | {entry['rows_before']:,} | {entry['rows_after']:,} | "
                     f"{dropped:,} ({share}) |")
    lines += [
        "",
        f"The reason for every one of those steps, written out as prose, is in "
        f"`{funnel.run_dir}/data_diary.md`. That file is the draft of the methods section.",
        "",
    ]
    return io.write_output_text(cfg, "task1_setup_note.md", "\n".join(lines))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--force", action="store_true",
                        help="rebuild everything, ignoring cached outputs")
    args = parser.parse_args()

    cfg = config_module.load(args.config)
    io.check_data_is_there(cfg)
    io.ensure_dirs(cfg)
    # Before anything is rewritten: say so if a report was edited by hand
    # since the last run. The stamped copy is authoritative, so this is a
    # warning and not an error (src/lib/io.py, and notes/decisions.md C-24).
    io.warn_about_hand_edited_outputs(cfg)
    funnel = Funnel("02_build_patent_tables", cfg)
    # Spill where the config says, under the memory ceiling the config sets.
    # This step is the heaviest in the project — 15 GB of bulk tables on an 8 GB
    # machine — and it was the ONLY step that set neither, so DuckDB used its
    # own defaults and wrote its spill into `.tmp/` in the repository root
    # instead of runtime.duckdb_temp_directory. One interrupted run left 11 GB
    # there. Step 03 has always done this correctly; step 02 did not, which is
    # the same defect class as C-38 and C-44 — a config key that looks obeyed
    # because the step you happen to read does obey it.
    Path(cfg.runtime.duckdb_temp_directory).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{cfg.runtime.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory = '{cfg.runtime.duckdb_temp_directory}'")
    if cfg.runtime.duckdb_threads:
        con.execute(f"SET threads = {cfg.runtime.duckdb_threads}")

    print(f"Task 1 — config '{cfg.name}' (hash {cfg.hash}), "
          f"sample_mode={'ON: CPC section ' + cfg.sample_cpc_section if cfg.sample_mode else 'off'}")

    # The pure config check this comment used to announce is not here and never
    # was: birth_years.gap_table_top_code having to exceed
    # birth_years.rule_b_tolerance_years is enforced by _check_consistency in
    # src/lib/config.py, which runs at config load above, before the connection
    # is even opened. That is earlier than here, which is what the check wanted.
    build_fields(con, cfg, funnel)
    patents = build_patents(con, cfg, funnel)
    patents_path = io.write_parquet(con, patents, out(cfg, "01_patents.parquet"), cfg,
                                    inputs=[raw(cfg, "patents"), raw(cfg, "application")],
                                    step="02_build_patent_tables")
    # Re-read from Parquet so later joins scan a materialised file rather than
    # re-running the whole filter chain for every query.
    io.read_parquet(con, patents_path).to_view("clean_patents", replace=True)

    field_rows = funnel.filter(
        con.sql("SELECT * FROM patent_fields_raw"), "fields_of_clean_patents",
        "patent_id IN (SELECT patent_id FROM clean_patents)",
        why=("Field assignments are restricted to the patents that survived the rules above, "
             "so that a field's size counts only patents this design can actually use."),
    )
    fields_path = io.write_parquet(con, field_rows, out(cfg, "01_patent_fields.parquet"), cfg,
                                   inputs=[raw(cfg, f"cpc_{cfg.cpc.table}")],
                                   step="02_build_patent_tables")
    io.read_parquet(con, fields_path).to_view("patent_fields_raw", replace=True)

    spine = build_inventor_patents(con, cfg, funnel)
    spine_path = io.write_parquet(con, spine, out(cfg, "01_inventor_patents.parquet"), cfg,
                                  inputs=[raw(cfg, "inventors"), patents_path],
                                  step="02_build_patent_tables")

    assignees = build_assignee_patents(con, cfg, funnel)
    assignees_path = io.write_parquet(con, assignees, out(cfg, "01_assignee_patents.parquet"),
                                      cfg, inputs=[raw(cfg, "assignees")],
                                      step="02_build_patent_tables")

    citations = build_citations(con, cfg, funnel)
    citations_path = io.write_parquet(con, citations, out(cfg, "01_citations.parquet"), cfg,
                                      inputs=[raw(cfg, "citations")],
                                      step="02_build_patent_tables")

    # Before the link is built, not after: if the persistent crosswalk's
    # 'current' column is a previous PatentsView release, every crosswalk link
    # made below would be to the wrong generation of identities.
    check_current_vintage_column(con, cfg, funnel)

    link = build_kjl_link(con, cfg, funnel)
    link_path = io.write_parquet(con, link, out(cfg, "01_kjl_link.parquet"), cfg,
                                 inputs=[kjl(cfg, "kjl_crosswalk"),
                                         raw(cfg, "persistent_inventor")],
                                         step="02_build_patent_tables")
    io.read_parquet(con, link_path).to_view("kjl_link_table", replace=True)

    # Run immediately after the link is built and before anything is measured on
    # it, so that a broken identity match stops the run rather than propagating
    # into birth years, deaths and every dominance share downstream.
    check_identity_match_by_name(con, cfg, funnel)

    # Deaths are resolved BEFORE birth years, not after, and the order is the
    # rule rather than a preference: Rule A takes a deceased inventor's birth
    # year from their death record, and Rule B tests that record against the
    # ages file, so neither can be computed until the records exist.
    resolve_death_records(con, cfg, funnel)

    births = build_birth_years(con, cfg, funnel)
    births_path = io.write_parquet(con, births, out(cfg, "01_inventor_birth_years.parquet"),
                                   cfg, inputs=[kjl(cfg, "kjl_ages"), kjl(cfg, "kjl_deaths"),
                                                link_path],
                                   step="02_build_patent_tables")
    io.read_parquet(con, births_path).to_view("birth_years_table", replace=True)

    deaths = build_deaths(con, cfg, funnel)
    deaths_path = io.write_parquet(con, deaths, out(cfg, "01_inventor_deaths.parquet"), cfg,
                                   inputs=[kjl(cfg, "kjl_deaths"), link_path, births_path],
                                   step="02_build_patent_tables")

    # The no-look-ahead guard is not needed for anything in Task 1, which
    # measures nothing as of a date. It is exercised here on the death table so
    # that a future change cannot quietly let a post-death year in.
    asof.check_no_future_years(
        io.read_parquet(con, deaths_path), cfg.deaths.year_max, ["death_year"],
        label="Task 1 death table",
    )

    # Read from the config rather than written out, so that a robustness run
    # dated by grant date, or one that keeps the incomplete recent years,
    # describes itself instead of repeating the default. The upper bound is
    # only a real bound when incomplete filing years are actually dropped.
    universe = cfg.patent_universe
    if universe.drop_incomplete_filing_years:
        years_kept = f"{universe.first_year}–{universe.filings_complete_through}"
    else:
        years_kept = f"{universe.first_year} onwards"

    produced = [
        ("patents",
         f"the patent universe, dated by {universe.date_by} date, {years_kept}",
         patents_path),
        ("fields", f"patent to field, at {cfg.fields.strategy}", fields_path),
        ("spine", "inventor x patent x field, with counting weights", spine_path),
        ("assignees", "patent to assignee", assignees_path),
        ("citations", "citations received by patents in this universe", citations_path),
        ("kjl link", "KJL identity to PatentsView identity", link_path),
        ("birth years", "the strict and loose lists", births_path),
        ("deaths", "one death per inventor, inside the window", deaths_path),
    ]
    stamped, plain = write_note(cfg, funnel, produced, con)
    io.write_latest(cfg, "02_build_patent_tables", [stamped],
                    summary="Task 1: the data set up and the basic rules fixed.")

    funnel.finish()
    print(f"\nNote: {plain}")


if __name__ == "__main__":
    main()
