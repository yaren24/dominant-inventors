"""
Task 2b item 8 — fractional shares sum to 1 within every field-year.

The advisor's words: "Add a test, if none exists, that fractional shares sum to 1
within every field-year. It is the cheapest guard against mixed counting." None
existed. Two places in src/ assert a sum-to-one property — 01_setup._check_weights
per patent and 02_dominance.validate() check #1 per field-year — and until this
file no test touched either, nor any other part of src/03_measure_dominance.py.

Three things here are worth more than the rest.

**Mixed counting IS expressible, against what 02_dominance's own docstring says.**
That docstring argues the check is nearly redundant: the denominator is a window
function over the numerator column, so "there is no other column in scope that
could be put there by mistake, so fractional-over-full and full-over-fractional
are not expressible." `windows_t` also carries `inventor_patents_in_window`, whose
name differs from `inventor_weight_in_window` by one word and whose window sum is
exactly the full-count denominator. Swapping them is a one-word edit, it type-checks,
it leaves the Task 1 weight reconciliation and the [0, 1] range check passing, and
the sum-to-one check is the only one of the nine that notices — measured, by
test_mixed_counting_is_expressible_and_only_this_check_notices below. So the
advisor's "cheapest guard" is not redundant with the structure; it is what makes
the structure safe. Structural arguments are worth what their verification is
worth, which is §8's closing rule.

Where a substitution genuinely is out of scope — the patent head count is in the
field-year table, not in `windows_t` — the failing arm rewrites the stored `share`
column after the real code has produced it, leaving every other column untouched.
That arm guards a future rewrite rather than today's code, and says so.

**The tolerance is not a formality and exact equality is not even a predicate.**
Measured 2026-08-05 at hash 80422f17: `sum(share) = 1.0` holds for 1.4% of
field-years at cpc_class, 7.7% at subclass, 30.0% at main group. Worse, eight runs
of one query on unchanged data gave seven different answers — DuckDB's parallel
hash aggregate fixes no summation order and floating-point addition is not
associative. So which rows `= 1.0` selects is not reproducible, and the tolerance
is the only defensible predicate. This inverts item 1's D1, where `= 1.0` and
`>= 1 - 1e-9` selected identical rows and the strict reading won.

**The check's power is in the `max`, not the tolerance.** The substitution
02_dominance warns about — dividing by the patent head count — leaves the median
field-year EXACTLY unchanged, because under fractional counting with primary codes
the field's weight total equals its head count in 74.8% to 98.9% of field-years.
It shows up only in the tail. A mean deviation would miss it.

See notes/decisions.md C-37 (D42-D48) and §16 of the item 2b design.
"""

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.lib import io
from src.lib.funnel import Funnel

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_step(name, filename):
    """A step file cannot be imported normally: a module name may not start with a digit."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_tables = load_step("build_patent_tables", "02_build_patent_tables.py")
dominance = load_step("measure_dominance", "03_measure_dominance.py")


# The deviation this tolerance absorbs is floating-point summation error, not an
# analytical choice — no result moves when it changes, which is why it is a
# constant here and not a config key (D42; a new key would force a 30-minute
# rebuild of steps 01 and 02 for data unchanged by construction).
#
# Worst deviation measured on the real tables, 2026-08-05 at hash 80422f17:
# 1.744e-12 at cpc_class over a field-year summing 301,327 terms. A share built
# from mixed counting misses by about 0.617. Any value in [1e-11, 1e-3] therefore
# gives the same verdict on every case measured, and the same one step 03 gives.
SHARE_SUM_TOLERANCE = 1e-9

# validate() returns a list of (name, passed, detail). Tests key on the name
# rather than the position, so inserting a check upstream does not silently
# repoint them at a different one.
SUM_CHECK = "shares sum to 1 within every field-year"
RANGE_CHECK = "every share lies in [0, 1]"
KEY_CHECK = "one row per inventor x field x year"

YEAR = 2000  # inside dominance.first_compute_year .. last_compute_year


# ---------------------------------------------------------------------------
# Fixtures: the smallest tables step 03 will run against
# ---------------------------------------------------------------------------

# The `dominance_config` factory these tests use lives in tests/conftest.py. It
# started here and moved when item 9 became the second file to need the same
# tables built under both counting methods; a fixture in a test module is not
# visible to another test module.


def solo_patents(n, field="A61K"):
    """n patents, one inventor each, all in one field and one year."""
    return [(f"P{i}", field, YEAR, [f"I{i}"]) for i in range(1, n + 1)]


def team_patents(n, team_size, prefix="T", field="A61K"):
    """
    n patents of `team_size` inventors each, nobody appearing twice.

    The prefix exists because two calls in one fixture must not collide: a patent
    id reused with a different team size would give the same person two different
    weights and quietly test something else.
    """
    return [(f"{prefix}{i}", field, YEAR, [f"{prefix}{i}_{s}" for s in range(team_size)])
            for i in range(1, n + 1)]


def write_task1_tables(cfg, patents):
    """
    The two Task 1 tables step 03 reads, plus the field map it would otherwise
    derive from the raw CPC tables.

    `patents` is a list of (patent_id, field_id, year, inventor_ids). An EMPTY
    inventor list is meaningful and used below: step 02 records that some patents
    carry no inventor record at all, and they are exactly what makes a field's
    patent head count differ from its weight total.

    01_patents.parquet is written with the columns step 03 reads rather than all
    seven — grant_date and filing_date are never touched downstream of Task 1.
    """
    io.ensure_dirs(cfg)
    spine_rows, patent_rows, field_rows = [], [], []

    for patent_id, field_id, year, inventors in patents:
        patent_rows.append((patent_id, "utility", year, year, year))
        field_rows.append((patent_id, field_id))
        for sequence, inventor_id in enumerate(inventors):
            # Exactly what src/02_build_patent_tables.py writes, both branches. Fractional
            # splits the patent across the NAMES on it; full gives each PERSON one
            # patent, so it divides by how many times that person is listed. A list
            # like ["I1", "I1", "I2"] therefore weights I1 at 1/3 + 1/3 under
            # fractional and 1/2 + 1/2 under full. Kept faithful because item 9
            # compares a derived full count against a real full build, and a fixture
            # that invented its own weights would compare nothing. See C-38.
            if cfg.counting.method == "fractional":
                weight = 1.0 / len(inventors)
            else:
                weight = 1.0 / inventors.count(inventor_id)
            spine_rows.append((patent_id, inventor_id, sequence, len(inventors),
                               field_id, year, year, year, weight))

    writer = duckdb.connect()
    writer.register("spine", pd.DataFrame(spine_rows, columns=[
        "patent_id", "inventor_id", "inventor_sequence", "team_size", "field_id",
        "application_year", "grant_year", "year", "weight"]))
    writer.register("patents", pd.DataFrame(patent_rows, columns=[
        "patent_id", "patent_type", "application_year", "grant_year", "year"]))
    writer.register("fields", pd.DataFrame(field_rows, columns=["patent_id", "field_id"]))

    for view, filename in [("spine", "01_inventor_patents.parquet"),
                           ("patents", "01_patents.parquet"),
                           ("fields", "test_patent_fields.parquet")]:
        destination = io.interim_path(cfg, filename).as_posix()
        writer.execute(f"COPY {view} TO '{destination}' (FORMAT parquet)")
    writer.close()


# ---------------------------------------------------------------------------
# Driving step 03's real functions
# ---------------------------------------------------------------------------

def stage_the_fixture_spine(con, cfg):
    """
    Stand in for build_spine_at_width by supplying its OUTPUT.

    That function assigns fields from the raw CPC tables and cannot run without
    3.8 GB of them, so the fixture supplies the five columns it produces, selected
    straight out of the synthetic Task 1 spine. That also makes
    check_matches_task_one_spine pass by construction rather than by luck — and the
    check is allowed to run, which is why the width is the config's own.

    THE SHORTCUT IS NOT FAITHFUL FOR DUPLICATE LISTINGS, and item 9 is where that
    matters. The real function collapses a person listed three times on one patent
    into ONE row carrying appearances = 3; this leaves three rows. Both give the
    same weight, so item 8's invariant cannot tell them apart — but
    `inventor_patents_in_window` comes out 3 here against 1 there, and that column
    is the full count item 9 derives from. Item 9 therefore drives the real
    function with fields.assign_fields patched; see tests/test_task2b_item9.py.
    """
    spine = io.interim_path(cfg, "01_inventor_patents.parquet").as_posix()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE spine_at_width AS
        SELECT patent_id, inventor_id, field_id, year AS filing_year, weight
        FROM read_parquet('{spine}')
    """)


def run_from_spine(con, cfg):
    """
    Everything from `spine_at_width` to validate(), using step 03's own functions.

    Split from run_real_chain() at item 9, which builds the spine with the real
    build_spine_at_width and needs the rest of the chain unchanged. The seam is a
    temp table both callers create, so neither has to know how the other filled it.

    Returns the checks by name, and the two paths so a test can rewrite them.
    """
    width = cfg.fields.strategy
    fields = io.interim_path(cfg, "test_patent_fields.parquet").as_posix()
    funnel = Funnel("test_shares_sum_to_one", cfg)

    # build_field_patent_counts() reads this view. It must carry the patents with
    # no inventor record too, or the head count would silently agree with the
    # weight total and the fixture would lose the thing it exists to create.
    con.sql(f"SELECT patent_id, field_id FROM read_parquet('{fields}')").to_view(
        "patent_fields_at_width", replace=True)

    spine_note = dominance.check_matches_task_one_spine(con, cfg, width)
    dominance.build_activity(con, cfg, funnel, width)
    windows = dominance.build_windows(con, cfg, funnel, width)
    windows.to_view("windows_source", replace=True)
    con.execute("CREATE OR REPLACE TEMP TABLE windows_t AS SELECT * FROM windows_source")

    inventor_path = dominance.write_inventor_field_years(con, cfg, width)
    field_path = dominance.write_field_years(con, cfg, width, inventor_path)
    results = dominance.validate(con, cfg, width, inventor_path, field_path, spine_note)
    funnel.finish(print_to_terminal=False)

    return by_name(results), inventor_path, field_path, spine_note


def run_real_chain(con, cfg):
    """The whole chain with the fixture standing in for build_spine_at_width."""
    stage_the_fixture_spine(con, cfg)
    return run_from_spine(con, cfg)


def by_name(results):
    return {name: (passed, detail) for name, passed, detail in results}


def revalidate_with_injected_shares(con, cfg, injected_sql, spine_note):
    """
    Replace the inventor table with `injected_sql` and run validate() again.

    This is how a mixed-counting share is produced at all (D44). Step 03's own
    SQL cannot express one, so the failing arm takes the table the real code just
    wrote and rewrites the single `share` column with the expression a future
    rewrite might put there. Every other column is left exactly as step 03 wrote
    it, so a failure is attributable to that one line and nothing else.

    The field-year table is rebuilt through the real write_field_years(), so the
    two files stay consistent with each other and check #7 keeps passing — a
    failing arm that tripped four unrelated checks would prove nothing about the
    one under test.
    """
    width = cfg.fields.strategy
    inventor_path = dominance.inventor_file(cfg, width)
    con.execute(f"CREATE OR REPLACE TEMP TABLE injected AS {injected_sql}")
    io.write_parquet(con, con.sql("SELECT * FROM injected"), inventor_path, cfg,
                     step="tests/test_shares_sum_to_one.py")
    field_path = dominance.write_field_years(con, cfg, width, inventor_path)
    return by_name(dominance.validate(con, cfg, width, inventor_path, field_path, spine_note))


def stored_inventor_table(cfg):
    return dominance.inventor_file(cfg, cfg.fields.strategy).as_posix()


@pytest.fixture
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Group 1 — the passing arm, through step 03's real SQL
# ---------------------------------------------------------------------------

def test_seven_solo_patents_sum_to_one_without_being_exactly_one(con, dominance_config):
    """
    The smallest case where the invariant holds and exact equality does not.

    Seven inventors each holding one seventh of a field. In IEEE double 1/7 added
    seven times is 0.9999999999999998, so this is the case that would fail a test
    written as `= 1.0` — the reason D43 uses a tolerance. On the real tables the
    smallest field-year that misses exact 1.0 has four inventors; none with three
    or fewer does.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, solo_patents(7))
    checks, inventor_path, _, _ = run_real_chain(con, cfg)

    passed, detail = checks[SUM_CHECK]
    assert passed, detail
    assert checks[RANGE_CHECK][0]
    assert checks[KEY_CHECK][0]

    # And the shares really are the coarse fractions, not a rounded stand-in.
    shares = con.execute(
        f"SELECT DISTINCT share FROM read_parquet('{inventor_path.as_posix()}')").fetchall()
    assert shares == [(1 / 7,)]


def test_teams_of_two_and_three_sum_to_one(con, dominance_config):
    """
    Fractional weights that are not all equal, and a 1/3 that has no exact binary
    representation. Item 2's finding was that fractional weights land on a lattice
    of unit fractions; this confirms the SUM of them still reaches 1, which is a
    different question — a lattice constrains values and says nothing about the
    order an aggregate adds them in.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, team_patents(4, 2, prefix="PAIR")
                       + team_patents(3, 3, prefix="TRIO"))
    checks, inventor_path, _, _ = run_real_chain(con, cfg)

    assert checks[SUM_CHECK][0], checks[SUM_CHECK][1]

    # 4 patents of 2 and 3 patents of 3 = 17 inventors, weight total 7.0.
    row = con.execute(f"""
        SELECT count(*), sum(inventor_weight_in_window), any_value(field_weight_in_window)
        FROM read_parquet('{inventor_path.as_posix()}') WHERE year = {YEAR}
    """).fetchone()
    assert row[0] == 17
    assert row[1] == pytest.approx(7.0)
    assert row[2] == pytest.approx(7.0)


def test_invariant_holds_under_full_counting_too(con, dominance_config):
    """
    The field-year invariant is ungated where the per-patent one is gated.

    01_setup._check_weights only runs under fractional + primary, because under
    full counting a patent's weights sum to its team size — measured on the real
    spine, 5,376,895 of 8,191,355 patents (65.6%) would fail it. The field-year
    check has no such gate and needs none: the denominator is the numerator
    summed, so the shares sum to 1 whatever the weights mean.
    """
    cfg = dominance_config(counting_method="full")
    write_task1_tables(cfg, team_patents(4, 2))
    checks, inventor_path, _, _ = run_real_chain(con, cfg)

    assert checks[SUM_CHECK][0], checks[SUM_CHECK][1]

    # Under full counting every inventor-patent is worth 1.0, so the field's
    # weight total is the number of pairs (8) rather than the number of patents (4).
    total = con.execute(f"""
        SELECT any_value(field_weight_in_window)
        FROM read_parquet('{inventor_path.as_posix()}') WHERE year = {YEAR}
    """).fetchone()[0]
    assert total == pytest.approx(8.0)


def test_the_spine_identity_check_actually_runs(con, dominance_config):
    """
    Guard the guard. check_matches_task_one_spine returns a cheerful string
    instead of comparing anything when the width is not the one Task 1 was built
    at, and validate() reports that as PASS — two of Task 2's 27 checks were
    marked PASS without running for exactly this reason. If this test ever starts
    seeing the skip message, the fixture stopped exercising the comparison.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, solo_patents(3))
    _, _, _, spine_note = run_real_chain(con, cfg)

    assert spine_note == "identical to Task 1, row for row"
    assert "not checked" not in spine_note


# ---------------------------------------------------------------------------
# Group 2 — the failing arm, injected per D44
# ---------------------------------------------------------------------------

CORRECT_DENOMINATOR = ("inventor_weight_in_window\n"
                       "                / sum(inventor_weight_in_window) "
                       "OVER (PARTITION BY field_id, year)\n"
                       "                AS share")
MIXED_DENOMINATOR = CORRECT_DENOMINATOR.replace("sum(inventor_weight_in_window)",
                                                "sum(inventor_patents_in_window)")


def test_mixed_counting_is_expressible_and_only_this_check_notices(con, dominance_config,
                                                                  monkeypatch):
    """
    The defect the advisor named, produced by step 03's own SQL rather than
    injected — and the test that makes this file worth having.

    02_dominance argues that mixed counting cannot be written by accident because
    the denominator is a window function over the numerator column. But
    `inventor_patents_in_window` is in scope in `windows_t` too, and the sum of it
    over the field-year IS the full-count denominator. So the substitution is a
    one-word edit to the real query, which is what this test performs.

    With four patents of two inventors the weight total is 4.0 and the pair count
    is 8, so the shares sum to 0.5 — a deviation eleven orders of magnitude above
    the largest legitimate one, so no choice of tolerance separates them wrongly.

    The three assertions at the end are the substance. The [0, 1] range check and
    the Task 1 weight reconciliation both still PASS under this mutation — the
    reconciliation because it compares weights and never looks at a share, the
    range check because 0.0625 is a perfectly ordinary number. The sum-to-one
    check is the only one of the nine that fails. That is the advisor's "cheapest
    guard" earning its place, measured rather than argued.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, team_patents(4, 2))

    original = dominance.add_share_rank_and_position
    assert CORRECT_DENOMINATOR in original(cfg), (
        "the share expression in 03_measure_dominance.py has been reformatted, so this "
        "mutation no longer applies and the test below proves nothing")
    monkeypatch.setattr(dominance, "add_share_rank_and_position",
                        lambda c: original(c).replace(CORRECT_DENOMINATOR, MIXED_DENOMINATOR))

    checks, _, _, _ = run_real_chain(con, cfg)

    assert not checks[SUM_CHECK][0]
    assert "5.000e-01" in checks[SUM_CHECK][1], checks[SUM_CHECK][1]
    assert checks[RANGE_CHECK][0], "the range check does not catch mixed counting"
    assert checks["total counting weight reconciles with Task 1"][0], (
        "the Task 1 reconciliation does not catch mixed counting")


def test_patent_head_count_as_denominator_is_caught(con, dominance_config):
    """
    The quiet twin, and the one 02_dominance's own docstring warns about: dividing
    by field_patents_in_window, the plain patent head count.

    It is quiet because under fractional counting with primary codes the head
    count usually EQUALS the weight total — measured on the real tables, in 74.8%
    of field-years at cpc_class, 91.5% at subclass and 98.9% at main group, with a
    median relative difference of exactly zero. A substitution invisible in the
    median field-year is caught only by an aggregate that looks at the worst one,
    which is why check #1 takes a max (D45).

    The fixture reproduces the real mechanism rather than inventing one: eight
    patents in the field, one of them carrying no inventor record, so the head
    count is 8 against a weight total of 7 and the shares sum to 7/8.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, solo_patents(7) + [("P_ORPHAN", "A61K", YEAR, [])])
    checks, _, field_path, spine_note = run_real_chain(con, cfg)
    assert checks[SUM_CHECK][0], "the fixture must be valid before it is broken"

    # The head count and the weight total genuinely differ here — otherwise the
    # substitution below would be undetectable and the test would prove nothing.
    head, weight = con.execute(f"""
        SELECT any_value(field_patents_in_window), any_value(field_weight_in_window)
        FROM read_parquet('{field_path.as_posix()}') WHERE year = {YEAR}
    """).fetchone()
    assert (head, weight) == (8, 7.0)

    checks = revalidate_with_injected_shares(con, cfg, f"""
        SELECT i.* REPLACE (
            i.inventor_weight_in_window / f.field_patents_in_window AS share)
        FROM read_parquet('{stored_inventor_table(cfg)}') i
        JOIN read_parquet('{field_path.as_posix()}') f USING (field_id, year)
    """, spine_note)

    passed, detail = checks[SUM_CHECK]
    assert not passed
    assert "1.250e-01" in detail, detail


def test_a_duplicated_inventor_row_is_caught(con, dominance_config):
    """
    Double counting from the other direction: one inventor's row present twice,
    so the field's shares sum to more than 1. Check #6 catches the duplicate key
    as well, and both are asserted — a test that only watched check #1 would not
    notice if the two ever stopped agreeing about the same table.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, solo_patents(4))
    _, _, _, spine_note = run_real_chain(con, cfg)

    checks = revalidate_with_injected_shares(con, cfg, f"""
        SELECT * FROM read_parquet('{stored_inventor_table(cfg)}')
        UNION ALL
        SELECT * FROM read_parquet('{stored_inventor_table(cfg)}') WHERE inventor_id = 'I1'
    """, spine_note)

    passed, detail = checks[SUM_CHECK]
    assert not passed
    assert "2.500e-01" in detail, detail          # one extra quarter share
    assert not checks[KEY_CHECK][0]


def test_an_empty_table_fails_rather_than_passes(con, dominance_config):
    """
    No field-years at all must be a failure, not a vacuous pass. `max()` over no
    rows returns NULL, and `worst < 1e-9` on NULL would be falsy for the wrong
    reason, so validate() guards it with `worst is not None`.

    That guard was unreachable until item 8 ran it: the detail string beside it
    formatted the same NULL with `:.3e` and raised TypeError first, so the check
    crashed instead of reporting. One line, found only by calling the function.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, solo_patents(4))
    _, _, _, spine_note = run_real_chain(con, cfg)

    checks = revalidate_with_injected_shares(con, cfg, f"""
        SELECT * FROM read_parquet('{stored_inventor_table(cfg)}') WHERE false
    """, spine_note)

    passed, detail = checks[SUM_CHECK]
    assert not passed
    assert "NO field-years" in detail, detail


def test_a_deviation_just_outside_the_tolerance_is_caught(con, dominance_config):
    """
    The boundary, from the failing side. Every share scaled by 1 + 2e-9, so the
    totals miss by twice the tolerance and nothing else about the table changes.

    This is the case that says the tolerance is a threshold rather than a
    decoration: the deviation here is 800 times smaller than the smallest real
    defect above, and it is still refused.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, solo_patents(7))
    _, _, _, spine_note = run_real_chain(con, cfg)

    checks = revalidate_with_injected_shares(con, cfg, f"""
        SELECT * REPLACE (share * (1.0 + 2e-9) AS share)
        FROM read_parquet('{stored_inventor_table(cfg)}')
    """, spine_note)

    passed, detail = checks[SUM_CHECK]
    assert not passed
    assert "2.000e-09" in detail, detail


# ---------------------------------------------------------------------------
# Group 3 — the tolerance itself
# ---------------------------------------------------------------------------

def test_the_tolerance_is_load_bearing_and_exact_equality_would_reject_valid_data(
        con, dominance_config):
    """
    The other side of the boundary, and the test that makes D43 a measurement.

    Shares scaled by 1 + 3e-10: inside the tolerance, so the check must pass —
    and provably not equal to 1.0, so a version of this check written as `= 1.0`
    would refuse a table it has no business refusing. Both halves are asserted
    together, because either alone is satisfiable by a broken comparison.

    The scaling factor is used rather than relying on a summation order to miss
    1.0 by itself. 3e-10 is 150 times the largest legitimate deviation ever
    measured (2.2e-12), so no thread count and no association order can put this
    total back on 1.0 — which matters, since eight runs of one query over the real
    tables gave seven different answers.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, solo_patents(7))
    _, _, _, spine_note = run_real_chain(con, cfg)

    checks = revalidate_with_injected_shares(con, cfg, f"""
        SELECT * REPLACE (share * (1.0 + 3e-10) AS share)
        FROM read_parquet('{stored_inventor_table(cfg)}')
    """, spine_note)

    passed, detail = checks[SUM_CHECK]
    assert passed, detail

    totals = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE total = 1.0),
               max(abs(total - 1.0))
        FROM (SELECT field_id, year, sum(share) AS total
              FROM read_parquet('{stored_inventor_table(cfg)}') GROUP BY 1, 2)
    """).fetchone()
    assert totals[0] > 0
    assert totals[1] == 0, "the fixture must not sum to exactly 1.0, or it tests nothing"
    assert 1e-10 < totals[2] < SHARE_SUM_TOLERANCE


def test_the_tolerance_here_matches_the_one_step_02_enforces(con, dominance_config):
    """
    The constant in this file is a copy of the one in 02_dominance.validate(), and
    a copy can drift. There is no config key to bind them (D42), so the binding is
    this: the two fixtures either side of SHARE_SUM_TOLERANCE must land on the
    same side of the step's own threshold as they do on ours.

    Not a strong test, and it is not pretending to be. It is the cheapest thing
    that fails if somebody changes 1e-9 in one file and not the other.
    """
    cfg = dominance_config()
    write_task1_tables(cfg, solo_patents(7))
    _, _, _, spine_note = run_real_chain(con, cfg)

    for factor, expected in [(1.0 + SHARE_SUM_TOLERANCE / 10, True),
                             (1.0 + SHARE_SUM_TOLERANCE * 10, False)]:
        checks = revalidate_with_injected_shares(con, cfg, f"""
            SELECT * REPLACE (share * {factor!r} AS share)
            FROM read_parquet('{stored_inventor_table(cfg)}')
        """, spine_note)
        passed, detail = checks[SUM_CHECK]
        assert passed is expected, f"factor {factor!r}: {detail}"


# ---------------------------------------------------------------------------
# Group 4 — the per-patent twin in step 02
# ---------------------------------------------------------------------------

def spine_relation(con, rows):
    """A relation shaped like the Task 1 spine, for _check_weights to inspect."""
    con.register("spine_rows", pd.DataFrame(rows, columns=["patent_id", "weight"]))
    return con.sql("SELECT patent_id, weight FROM spine_rows")


def test_check_weights_accepts_fractional_weights_that_sum_to_one(con, dominance_config):
    cfg = dominance_config()
    spine = spine_relation(con, [("P1", 0.5), ("P1", 0.5), ("P2", 1 / 3),
                                 ("P2", 1 / 3), ("P2", 1 / 3)])
    build_tables._check_weights(con, spine, cfg)  # must not raise


def test_check_weights_refuses_weights_that_do_not_sum_to_one(con, dominance_config):
    """
    The per-patent half of the same invariant, and the arm that proves the check
    can fail: two inventors each credited with half of half a patent.
    """
    cfg = dominance_config()
    spine = spine_relation(con, [("P1", 0.25), ("P1", 0.25)])
    with pytest.raises(ValueError, match="do not sum to 1"):
        build_tables._check_weights(con, spine, cfg)


def test_check_weights_refuses_a_weight_outside_the_unit_interval(con, dominance_config):
    cfg = dominance_config()
    spine = spine_relation(con, [("P1", 0.0), ("P1", 1.0)])
    with pytest.raises(ValueError, match="outside"):
        build_tables._check_weights(con, spine, cfg)


def test_check_weights_is_skipped_under_full_counting(con, dominance_config):
    """
    Pinning the gate, not the check. Under full counting a patent's weights sum
    to its team size, so the per-patent test is not merely unnecessary — it is
    false for 65.6% of the real spine, and "fixing" the gate would break every
    full-counting run. The range check still applies and still runs.
    """
    cfg = dominance_config(counting_method="full")
    spine = spine_relation(con, [("P1", 1.0), ("P1", 1.0), ("P1", 1.0)])
    build_tables._check_weights(con, spine, cfg)  # sums to 3.0, and that is correct here

    with pytest.raises(ValueError, match="outside"):
        build_tables._check_weights(con, spine_relation(con, [("P1", 1.5)]), cfg)
