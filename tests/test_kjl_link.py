"""
Tests for the second KJL -> PatentsView identity link, the one built from
PatentsView's own crosswalk between disambiguation releases.

These run the real functions from src/02_build_patent_tables.py against tiny synthetic
tables, rather than re-implementing the SQL in Python. There is one definition
of the link and these tests exercise the same one the step uses.

Two things here are worth more than the rest.

The first is test_wrong_release_column_is_refused. Naming the wrong release
column is the mistake this whole design is exposed to, and it is a quiet one:
measured against the real data, the releases either side of the right one still
resolve about 91% of KJL's identities. A wrong answer looks like a slightly
disappointing right answer. Only a check that fails hard prevents it.

The second is that a 2018 identity can map to SEVERAL current inventors. The
link picks one, so the tests pin down exactly which one and confirm that two
runs over the same data always pick the same one.
"""

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.lib.funnel import Funnel

REPO_ROOT = Path(__file__).resolve().parents[1]

# The step is '02_build_patent_tables.py', which cannot be written as a normal import because
# a module name may not start with a digit. Loading it by path is the only way.
_spec = importlib.util.spec_from_file_location(
    "build_patent_tables", REPO_ROOT / "src" / "02_build_patent_tables.py"
)
step = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step)

KJL_COLUMN = "disamb_inventor_id_20181127"
NOW_COLUMN = "disamb_inventor_id_20251231"


# ---------------------------------------------------------------------------
# Building a miniature version of the three tables the link reads
# ---------------------------------------------------------------------------

def write_tables(tmp_path, persistent_rows, kjl_ids, inventor_rows=None):
    """
    Lay out the smallest set of files the link functions can run against.

    `persistent_rows` are (patent_id, sequence, id_in_kjl_release, id_now)
    tuples, standing in for PatentsView's crosswalk between releases. An empty
    string means that release did not know this inventor slot.
    """
    if inventor_rows is None:
        # By default the inventor table simply agrees with the crosswalk's
        # current column, which is the situation on the real data.
        inventor_rows = [(p, s, now) for p, s, _, now in persistent_rows if now]

    # Written with DuckDB rather than pandas because pandas needs pyarrow for
    # Parquet and DuckDB is already the repository's Parquet writer. Each table
    # is registered under a name first, so the SQL below refers to something
    # that was explicitly handed to it.
    writer = duckdb.connect()
    writer.register("persistent", pd.DataFrame(
        persistent_rows, columns=["patent_id", "inventor_sequence", KJL_COLUMN, NOW_COLUMN]))
    writer.register("inventors", pd.DataFrame(
        inventor_rows, columns=["patent_id", "inventor_sequence", "inventor_id"]))
    writer.register("crosswalk", pd.DataFrame(
        [(i, "1") for i in kjl_ids], columns=["inventor_id", "patent_id"]))

    writer.execute(f"COPY persistent TO '{tmp_path / 'persistent.parquet'}' (FORMAT parquet)")
    writer.execute(f"COPY inventors  TO '{tmp_path / 'inventors.parquet'}' (FORMAT parquet)")
    writer.execute(f"COPY crosswalk  TO '{tmp_path / 'kjl_crosswalk.csv'}' (FORMAT csv, HEADER)")
    writer.close()


@pytest.fixture
def make_link_config(base_values, make_config, tmp_path):
    """A config pointing the three table names at the miniature files above."""
    def build(**overrides):
        values = base_values
        values["paths"]["interim"] = str(tmp_path)
        values["paths"]["raw_kjl"] = str(tmp_path)
        values["paths"]["logs"] = str(tmp_path / "runs")
        values["paths"]["outputs"] = str(tmp_path / "outputs")
        values["tables"]["persistent_inventor"] = "persistent.parquet"
        values["tables"]["inventors"] = "inventors.parquet"
        values["tables"]["kjl_crosswalk"] = "kjl_crosswalk.csv"
        values["columns"]["persistent_kjl_vintage"] = overrides.get("kjl_column", KJL_COLUMN)
        values["columns"]["persistent_current_vintage"] = overrides.get("now_column", NOW_COLUMN)
        return make_config(values)
    return build


def run_crosswalk(cfg):
    """Build the link and return it as a DataFrame, keyed by KJL identity."""
    con = duckdb.connect()
    funnel = Funnel("test_kjl_link", cfg)
    step.build_kjl_link_via_crosswalk(con, cfg, funnel)
    return con.sql("SELECT * FROM kjl_crosswalk_link ORDER BY kjl_inventor_id").df()


# ---------------------------------------------------------------------------
# The ordinary case: one person, one identity, before and after
# ---------------------------------------------------------------------------

def test_an_unambiguous_identity_links_to_one_current_inventor(make_link_config, tmp_path):
    write_tables(
        tmp_path,
        persistent_rows=[
            ("100", 0, "100-1", "fl:jo_ln:smith-1"),
            ("200", 0, "100-1", "fl:jo_ln:smith-1"),
        ],
        kjl_ids=["100-1"],
    )
    link = run_crosswalk(make_link_config())

    assert len(link) == 1
    row = link.iloc[0]
    assert row.crosswalk_inventor_id == "fl:jo_ln:smith-1"
    assert row.crosswalk_n_current_ids == 1
    assert row.crosswalk_patent_count == 2
    assert row.crosswalk_primary_share == 1.0


# ---------------------------------------------------------------------------
# The case the design has to make a choice about: the person was split
# ---------------------------------------------------------------------------

def test_a_split_person_links_to_whoever_holds_most_of_the_patents(make_link_config, tmp_path):
    """
    Three of the 2018 person's four patents are now one inventor and one is
    another. The link takes the majority holder and records that the split
    happened, so the ambiguity survives into the sample-selection step.
    """
    write_tables(
        tmp_path,
        persistent_rows=[
            ("100", 0, "100-1", "fl:jo_ln:smith-1"),
            ("200", 0, "100-1", "fl:jo_ln:smith-1"),
            ("300", 0, "100-1", "fl:jo_ln:smith-1"),
            ("400", 0, "100-1", "fl:jo_ln:smith-9"),
        ],
        kjl_ids=["100-1"],
    )
    row = run_crosswalk(make_link_config()).iloc[0]

    assert row.crosswalk_inventor_id == "fl:jo_ln:smith-1"
    assert row.crosswalk_n_current_ids == 2
    assert row.crosswalk_patent_count == 4
    assert row.crosswalk_primary_share == 0.75


def test_an_even_split_is_broken_the_same_way_every_run(make_link_config, tmp_path):
    """
    Two current inventors hold one patent each, so patent count cannot decide.
    The tie breaks on the identifier, which means the answer does not depend on
    the order DuckDB happened to scan the file in. Without this the link would
    not be reproducible, and a rerun could silently move the person.
    """
    write_tables(
        tmp_path,
        persistent_rows=[
            ("100", 0, "100-1", "fl:jo_ln:smith-9"),
            ("200", 0, "100-1", "fl:jo_ln:smith-1"),
        ],
        kjl_ids=["100-1"],
    )
    cfg = make_link_config()
    first = run_crosswalk(cfg).iloc[0].crosswalk_inventor_id

    assert first == "fl:jo_ln:smith-1"          # the lower identifier, not the first row
    assert run_crosswalk(cfg).iloc[0].crosswalk_inventor_id == first
    assert run_crosswalk(cfg).iloc[0].crosswalk_primary_share == 0.5


# ---------------------------------------------------------------------------
# Blanks are absent releases, not people
# ---------------------------------------------------------------------------

def test_a_blank_release_cell_is_not_treated_as_an_inventor(make_link_config, tmp_path):
    """
    A patent granted after a release has no identity in that release's column.
    Counting the blank as a person would invent an inventor whose identifier is
    the empty string and hand every such patent to them.
    """
    write_tables(
        tmp_path,
        persistent_rows=[
            ("100", 0, "100-1", "fl:jo_ln:smith-1"),
            ("900", 0, "",      "fl:ne_ln:person-1"),   # granted after the release
            ("901", 0, "100-1", ""),                    # slot no longer in the current data
        ],
        kjl_ids=["100-1"],
    )
    link = run_crosswalk(make_link_config())

    assert list(link.kjl_inventor_id) == ["100-1"]
    assert link.iloc[0].crosswalk_patent_count == 1     # 901 excluded, not counted
    assert "" not in set(link.crosswalk_inventor_id)


def test_an_identity_the_release_never_knew_is_absent_rather_than_wrong(
    make_link_config, tmp_path
):
    """
    The real case is the 990 KJL identities anchored on a Statutory Invention
    Registration, which no early PatentsView disambiguation covered. They must
    come back with no answer, so the link table can hold a null, rather than
    being attached to somebody.

    The surrounding 200 ordinary identities are not padding. Coverage is
    checked against a floor, so an unresolvable identity is only tolerable as a
    small minority — which is exactly its status on the real data, 990 of 1.9
    million. A fixture of one good and one bad identity would be a 50% failure.
    """
    ordinary = [str(n) for n in range(100, 300)]
    write_tables(
        tmp_path,
        persistent_rows=[(p, 0, f"{p}-1", f"fl:jo_ln:smith-{p}") for p in ordinary],
        kjl_ids=[f"{p}-1" for p in ordinary] + ["H0018384-1"],
    )
    link = run_crosswalk(make_link_config())

    assert "H0018384-1" not in set(link.kjl_inventor_id)
    assert len(link) == len(ordinary)


def test_a_padded_sir_identity_still_matches_the_short_form(make_link_config, tmp_path):
    """
    KJL writes SIR numbers as 'H' plus six digits plus a check character;
    PatentsView writes them short. H0018384 is SIR H1838. If a release ever
    does label someone with a SIR, the two spellings have to meet — the same
    repair C-21 added for the coordinate method.
    """
    write_tables(
        tmp_path,
        persistent_rows=[("H1838", 0, "H1838-1", "fl:jo_ln:smith-1")],
        kjl_ids=["H0018384-1"],
    )
    row = run_crosswalk(make_link_config()).iloc[0]

    assert row.kjl_inventor_id == "H0018384-1"          # reported as KJL writes it
    assert row.crosswalk_inventor_id == "fl:jo_ln:smith-1"


# ---------------------------------------------------------------------------
# The tripwires
# ---------------------------------------------------------------------------

def test_the_current_release_column_is_confirmed_against_the_inventor_table(
    make_link_config, tmp_path
):
    write_tables(
        tmp_path,
        persistent_rows=[
            ("100", 0, "100-1", "fl:jo_ln:smith-1"),
            ("200", 0, "100-1", "fl:jo_ln:smith-1"),
        ],
        kjl_ids=["100-1"],
    )
    cfg = make_link_config()
    con, funnel = duckdb.connect(), Funnel("test_kjl_link", cfg)

    assert step.check_current_vintage_column(con, cfg, funnel) == 1.0


def test_wrong_release_column_is_refused(make_link_config, tmp_path):
    """
    The column named as 'current' holds an older generation of identities. On
    the real data this mistake still resolves about 91% of everything, so it
    has to fail loudly rather than be judged by eye.
    """
    write_tables(
        tmp_path,
        persistent_rows=[
            ("100", 0, "100-1", "fl:jo_ln:smith-1"),
            ("200", 0, "100-1", "fl:jo_ln:smith-1"),
        ],
        kjl_ids=["100-1"],
    )
    cfg = make_link_config(now_column=KJL_COLUMN)      # point 'current' at the old release
    con, funnel = duckdb.connect(), Funnel("test_kjl_link", cfg)

    with pytest.raises(ValueError, match="DIFFERENT disambiguation release"):
        step.check_current_vintage_column(con, cfg, funnel)


def test_a_column_that_does_not_exist_is_named_in_the_error(make_link_config, tmp_path):
    write_tables(
        tmp_path,
        persistent_rows=[("100", 0, "100-1", "fl:jo_ln:smith-1")],
        kjl_ids=["100-1"],
    )
    cfg = make_link_config(kjl_column="disamb_inventor_id_19990101")
    con, funnel = duckdb.connect(), Funnel("test_kjl_link", cfg)

    with pytest.raises(ValueError, match="disamb_inventor_id_19990101"):
        step.check_current_vintage_column(con, cfg, funnel)


def test_a_release_that_resolves_almost_nothing_is_refused(make_link_config, tmp_path):
    """
    The coverage floor. If the named release does not contain KJL's identities,
    the link is being made to the wrong generation of people and the run stops.
    """
    write_tables(
        tmp_path,
        persistent_rows=[("100", 0, "100-1", "fl:jo_ln:smith-1")],
        kjl_ids=["100-1"] + [f"{n}-1" for n in range(500, 800)],
    )
    cfg = make_link_config()
    con, funnel = duckdb.connect(), Funnel("test_kjl_link", cfg)

    with pytest.raises(ValueError, match="do not lower this floor"):
        step.build_kjl_link_via_crosswalk(con, cfg, funnel)


# ---------------------------------------------------------------------------
# Which route owns an identity when the two disagree (PREFERRED_ROUTE, C-23)
# ---------------------------------------------------------------------------

def owner_of(rows):
    """Run the step's own `owner_inventor_id_sql()` over a tiny link table."""
    con = duckdb.connect()
    con.register("link", pd.DataFrame(rows))
    return con.execute(f"""
        SELECT a.kjl_inventor_id, {step.owner_inventor_id_sql()} AS owner_inventor_id
        FROM link a JOIN link x ON x.kjl_inventor_id = a.kjl_inventor_id
        ORDER BY a.kjl_inventor_id
    """).df()


def test_the_crosswalk_owns_an_identity_the_two_routes_disagree_about():
    """
    The decision of 2026-09-07, exercised on the step's own SQL rather than a copy.

    The two routes name different people for 27,255 of 1,858,356 KJL identities.
    Whichever wins decides who a death belongs to, and both answers are real people
    with real patents, so getting it wrong raises nothing — it moves an event from
    one inventor to another and every count stays plausible. Measured before the
    decision: the coordinate route gives 3,251 people at subclass an in-window death
    and the crosswalk 3,287, and every family x width x arm cell moved the same way.
    """
    owners = owner_of([
        {"kjl_inventor_id": "1-1", "patentsview_inventor_id": "coord_says",
         "crosswalk_inventor_id": "xw_says"},
        {"kjl_inventor_id": "2-1", "patentsview_inventor_id": "agreed",
         "crosswalk_inventor_id": "agreed"},
    ])

    assert list(owners["owner_inventor_id"]) == ["xw_says", "agreed"]


def test_an_identity_the_crosswalk_cannot_place_keeps_its_coordinate_owner():
    """
    The case the rule must not break: 830 identities anchored on a SIR.

    PatentsView's pre-2020 disambiguations never covered Statutory Invention
    Registrations, so those identities have a blank crosswalk column and always
    will. Preferring the crosswalk without a fallback would silently drop every
    death attached to them, which is 830 people the coordinate route places fine.
    """
    owners = owner_of([
        {"kjl_inventor_id": "H1-1", "patentsview_inventor_id": "sir_person",
         "crosswalk_inventor_id": None},
    ])

    assert list(owners["owner_inventor_id"]) == ["sir_person"]


def test_a_route_nobody_recognises_stops_the_run(monkeypatch):
    """
    A typo in the route name must not fall through to some default.

    It decides which person every death in the design is attached to. There is no
    safe guess, so an unrecognised value raises rather than picking one.
    """
    monkeypatch.setattr(step, "PREFERRED_ROUTE", "corsswalk")

    with pytest.raises(ValueError, match="neither 'crosswalk' nor 'coordinate'"):
        step.owner_inventor_id_sql()
