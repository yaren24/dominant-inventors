"""
Repairing one identifier format difference between KJL and PatentsView.

Why this file exists
--------------------
The KJL crosswalk and PatentsView disagree about how to write the number of a
Statutory Invention Registration, and they disagree in a way that produces no
error and no warning — only rows that quietly fail to join.

KJL writes a SIR as 'H', six digits, and one CHECK CHARACTER:

    H0000019   is SIR H1        H0000027   is SIR H2
    H0000035   is SIR H3        H0000043   is SIR H4

The eighth character is a checksum over the preceding digits, not part of the
number. That is why the trailing digits run 9, 7, 5, 3 while the patents they
name run 1, 2, 3, 4. Sixteen of these identifiers carry '&' as the check
character rather than a digit.

PatentsView writes the same four patents as 'H1', 'H2', 'H3', 'H4'.

Left unrepaired this costs more than the 1,810 patents that fail to join.
A KJL inventor identifier names a person by one appearance — 'H0000019-1' is
inventor position 1 on that patent — so an inventor anchored on a SIR cannot be
resolved at all, and 830 people with death records were being dropped for a
reason recorded in the data diary as 'the patent was withdrawn or renumbered',
which was not true. A further 2,833 inventors were recorded as disagreeing
across disambiguation vintages purely because a SIR in their KJL patent set
could never be matched to the same SIR in PatentsView.

Why the repair is deliberately narrow
-------------------------------------
It touches ONLY eight-character H identifiers. Everything else is returned
unchanged, byte for byte.

That restraint is the point. A general normaliser — one that stripped zero
padding from every prefixed identifier — would rewrite roughly 380,000 D, PP,
RE and T identifiers that already join correctly today. A failed join is
visible: it shows up as a missing row. A rewrite that happens to collide with a
different real patent number is invisible: it shows up as a number that is
merely wrong. This function cannot create a false match, because it cannot
alter any identifier that already matches.

PatentsView is itself inconsistent here — it stores H1 through H2007 unpadded
but H002008 through H002294 zero-padded to six digits. The crosswalk stores that
same high block in the same padded seven-character form, which is why 261 SIRs
matched before this repair existed and 1,810 did not.
"""

# One definition, used by the steps and exercised directly by the tests, so the
# rule cannot drift between what is tested and what runs.
#
# TRY_CAST rather than CAST: an eight-character H identifier whose middle six
# characters are not digits is not something we have ever seen, and it must not
# be guessed at. It becomes NULL here and `check_every_id_repaired` below turns
# that NULL into a loud failure rather than a silently dropped row.
REPAIR_MACRO = """
    CREATE OR REPLACE MACRO normalise_crosswalk_patent_id(raw_id) AS
        CASE
            WHEN raw_id LIKE 'H%' AND length(raw_id) = 8
            THEN 'H' || CAST(TRY_CAST(substr(raw_id, 2, 6) AS BIGINT) AS VARCHAR)
            ELSE raw_id
        END
"""


# The switched-off form of the same macro. It exists so that the queries in the
# steps are written once and read the same either way: turning the repair off is
# a config change, never a different code path.
PASS_THROUGH_MACRO = """
    CREATE OR REPLACE MACRO normalise_crosswalk_patent_id(raw_id) AS raw_id
"""


def register(con, repair=True):
    """
    Make normalise_crosswalk_patent_id() available to SQL on this connection.

    Call once per connection, before any query that reads the KJL crosswalk.
    With repair=False the macro returns its argument untouched, which reproduces
    the behaviour from before this repair existed.
    """
    con.execute(REPAIR_MACRO if repair else PASS_THROUGH_MACRO)


def check_every_id_repaired(con, relation, column, label):
    """
    Raise if the repair turned any identifier into NULL.

    A NULL here means an eight-character H identifier that did not hold six
    digits where six digits were expected — a format this code does not
    understand. Without this check such a row would simply fail to join and be
    dropped by the next funnel filter, which is exactly the kind of silent loss
    the repair exists to end.
    """
    relation.to_view("ids_being_checked", replace=True)
    unrepaired = con.sql(
        f"SELECT count(*) FROM ids_being_checked WHERE {column} IS NULL"
    ).fetchone()[0]

    if unrepaired:
        examples = con.sql(
            f"SELECT * FROM ids_being_checked WHERE {column} IS NULL LIMIT 5"
        ).df().to_string(index=False)
        raise ValueError(
            f"{label}: {unrepaired:,} patent identifiers could not be repaired.\n"
            f"These look like KJL's eight-character check-digit form but do not hold six "
            f"digits after the 'H'. Look at them before deciding what they are — do not "
            f"widen the rule in src/lib/patent_ids.py to make them pass.\n\n{examples}"
        )
