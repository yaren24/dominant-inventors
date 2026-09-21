"""
Tests for the KJL / PatentsView identifier repair.

The important tests in this file are the ones that confirm the repair does
NOTHING to identifiers that already join. The failure this repair fixes was
silent — 1,810 patents that simply did not match — and the way to turn a silent
missing row into a silent wrong row is to normalise too eagerly and land on a
different real patent number. So the pass-through cases below are not padding;
they are the guard rail.

The macro is tested by running it, rather than by re-implementing the rule in
Python and testing that. There is one definition of the repair and these tests
exercise the same one the steps use.
"""

import duckdb
import pandas as pd
import pytest

from src.lib import patent_ids


@pytest.fixture
def con():
    connection = duckdb.connect()
    patent_ids.register(connection)
    return connection


def repair(con, raw_id):
    """Run one identifier through the macro, exactly as the steps do."""
    return con.execute(
        "SELECT normalise_crosswalk_patent_id(?)", [raw_id]
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# The case the repair exists for
# ---------------------------------------------------------------------------

def test_check_digit_form_becomes_the_patentsview_form(con):
    """
    KJL writes 'H' + six digits + one check character. The check character is a
    checksum, not part of the number, which is why these four run 9, 7, 5, 3
    while the patents they name run 1, 2, 3, 4.
    """
    assert repair(con, "H0000019") == "H1"
    assert repair(con, "H0000027") == "H2"
    assert repair(con, "H0000035") == "H3"
    assert repair(con, "H0000043") == "H4"


def test_check_character_may_be_punctuation(con):
    """
    Sixteen identifiers in the crosswalk carry '&' where a digit belongs.

    That H001835& is SIR H1835 is not a guess: in the crosswalk it sits inside
    the unbroken run H0018309, H0018317, H0018325 ... H0018406, which repairs to
    H1830, H1831, H1832 ... H1840. Only the check character is unusual.
    """
    assert repair(con, "H001835&") == "H1835"


def test_four_digit_sir_survives_the_repair(con):
    """Note H0018384, not H0001838 — six digits of number, then the check digit."""
    assert repair(con, "H0018384") == "H1838"


def test_the_top_of_the_range_lands_where_patentsview_stops_padding(con):
    """
    The strongest single piece of evidence that the format has been read
    correctly. The largest check-digit identifier in the crosswalk is H0020079,
    and PatentsView's unpadded H series ends at exactly H2007 before switching
    to the padded H002008 form. The two conventions meet precisely.
    """
    assert repair(con, "H0020079") == "H2007"


# ---------------------------------------------------------------------------
# Everything the repair must leave alone
# ---------------------------------------------------------------------------

def test_utility_numbers_are_untouched(con):
    """
    The overwhelming majority of the crosswalk. Seven-digit and eight-digit
    utility numbers already join; rewriting one could only ever break it.
    """
    assert repair(con, "3930273") == "3930273"
    assert repair(con, "10000000") == "10000000"


def test_other_prefixes_are_untouched(con):
    """
    D, PP, RE and T identifiers were verified to join correctly as they stand —
    all 383,000 of them agree with PatentsView on grant year. A general
    normaliser would rewrite every one of these for no reason.
    """
    assert repair(con, "D242584") == "D242584"
    assert repair(con, "PP3987") == "PP3987"
    assert repair(con, "RE28671") == "RE28671"
    assert repair(con, "T100201") == "T100201"


def test_the_seven_character_sir_form_is_untouched(con):
    """
    PatentsView is internally inconsistent: it writes H1 through H2007 unpadded
    but H002008 through H002294 padded. The crosswalk writes that high block the
    same padded way, so those 261 already match and must not be 'fixed'.
    """
    assert repair(con, "H002008") == "H002008"
    assert repair(con, "H002294") == "H002294"


def test_short_h_identifiers_are_untouched(con):
    """The PatentsView form itself, in case it is ever fed back through."""
    assert repair(con, "H1") == "H1"
    assert repair(con, "H1838") == "H1838"


# ---------------------------------------------------------------------------
# Switched off
# ---------------------------------------------------------------------------

def test_repair_can_be_switched_off_from_config():
    """
    kjl_link.repair_sir_patent_ids: false must reproduce the pre-repair
    behaviour exactly, so the old numbers can be regenerated on demand.
    """
    connection = duckdb.connect()
    patent_ids.register(connection, repair=False)
    assert repair(connection, "H0000019") == "H0000019"
    assert repair(connection, "3930273") == "3930273"


# ---------------------------------------------------------------------------
# Failing loudly on a form this code does not understand
# ---------------------------------------------------------------------------

def test_unparseable_h_identifier_raises_rather_than_being_dropped(con):
    """
    An eight-character H identifier without six digits is a format nobody has
    seen. It must stop the run, not vanish into the next funnel filter — being
    silently dropped is the exact failure this whole module exists to end.
    """
    frame = pd.DataFrame({"anchor_patent_id": ["H1", None, "3930273"]})
    con.register("frame", frame)
    relation = con.sql("SELECT * FROM frame")

    with pytest.raises(ValueError, match="could not be repaired"):
        patent_ids.check_every_id_repaired(
            con, relation, "anchor_patent_id", label="test identifiers",
        )


def test_fully_repaired_identifiers_pass_the_check(con):
    frame = pd.DataFrame({"anchor_patent_id": ["H1", "H1838", "3930273"]})
    con.register("frame", frame)
    relation = con.sql("SELECT * FROM frame")

    patent_ids.check_every_id_repaired(
        con, relation, "anchor_patent_id", label="test identifiers",
    )
