"""
Tests for the CPC field strategies, and in particular for the guard that stops a
run whose primary-code rule matches nothing.

`cpc.primary_sequence` and `cpc.table` look like two independent settings and
are not: g_cpc_current numbers its codes from 0, g_cpc_at_issue from 1. Setting
the table alone leaves the primary-code filter selecting no rows, every patent
loses its field, and — before this guard — the run failed several joins later
with an empty universe, which reads like a problem with the data rather than
with the config. Found at Task 2b item 5; see notes/decisions.md C-33.
"""

import duckdb
import pandas as pd
import pytest

from src.lib.field_strategies import cpc


def cpc_table(tmp_path, rows):
    """
    Write a raw CPC table from (patent_id, sequence, subclass, group) tuples.

    Everything is text, as step 01 writes it: PatentsView TSVs carry no types,
    and the casts in cpc.py are part of what these tests exercise.
    """
    frame = pd.DataFrame(  # noqa: F841 — read by DuckDB's replacement scan
        [{"patent_id": p, "cpc_sequence": s, "cpc_section": subclass[:1],
          "cpc_class": subclass[:3], "cpc_subclass": subclass, "cpc_group": group}
         for p, s, subclass, group in rows],
        columns=["patent_id", "cpc_sequence", "cpc_section", "cpc_class",
                 "cpc_subclass", "cpc_group"])
    path = tmp_path / "cpc.parquet"
    duckdb.connect().execute(f"COPY (SELECT * FROM frame) TO '{path}' (FORMAT PARQUET)")
    return path


def cpc_config(base_values, make_config, tmp_path):
    base_values["paths"]["interim"] = str(tmp_path)
    base_values["tables"]["cpc_current"] = "cpc.parquet"
    return make_config(base_values)


def test_the_primary_rule_keeps_one_row_per_patent(base_values, make_config, tmp_path):
    """The ordinary case: sequence 0 exists, and it is the row that is kept."""
    cpc_table(tmp_path, [("p1", "0", "A61K", "A61K31/00"),
                         ("p1", "1", "H01L", "H01L21/00"),
                         ("p2", "0", "H01L", "H01L21/00")])
    cfg = cpc_config(base_values, make_config, tmp_path)

    fields = cpc.cpc_subclass(duckdb.connect(), cfg).df()

    assert fields.patent_id.tolist() == ["p1", "p2"]
    assert fields.field_id.tolist() == ["A61K", "H01L"]


def test_a_table_numbered_from_one_stops_the_run_and_names_both_settings(
        base_values, make_config, tmp_path):
    """
    The g_cpc_at_issue case. Every code is there, every patent is classified, and
    the configured sequence selects none of them.

    The message has to name the value that would work, because the failure gives
    no hint on its own: an empty result from a full table looks like bad data.
    """
    cpc_table(tmp_path, [("p1", "1", "A61K", "A61K31/00"),
                         ("p1", "2", "H01L", "H01L21/00"),
                         ("p2", "1", "H01L", "H01L21/00")])
    cfg = cpc_config(base_values, make_config, tmp_path)
    assert cfg.cpc.primary_sequence == 0        # the setting under test

    with pytest.raises(ValueError, match="matches no rows") as raised:
        cpc.cpc_subclass(duckdb.connect(), cfg)

    message = str(raised.value)
    assert "lowest sequence is 1" in message
    assert "cpc.primary_sequence" in message


def test_an_empty_cpc_table_says_so_rather_than_blaming_the_sequence(
        base_values, make_config, tmp_path):
    """
    Nothing before the rule ran is a different fault from nothing after it, and
    telling the two apart is the whole point of a diagnostic: one is a missing
    or mis-pathed file, the other is a setting.
    """
    cpc_table(tmp_path, [])
    cfg = cpc_config(base_values, make_config, tmp_path)

    with pytest.raises(ValueError, match="No CPC rows at all"):
        cpc.cpc_subclass(duckdb.connect(), cfg)


def test_the_all_codes_alternative_is_not_touched_by_the_guard(
        base_values, make_config, tmp_path):
    """
    With `cpc.assignment: all` there is no primary-code rule, so a table numbered
    from 1 is perfectly usable and must not raise. The guard checks a rule that
    was applied, never a rule that was skipped.
    """
    cpc_table(tmp_path, [("p1", "1", "A61K", "A61K31/00"),
                         ("p1", "2", "H01L", "H01L21/00")])
    base_values["cpc"]["assignment"] = "all"
    cfg = cpc_config(base_values, make_config, tmp_path)

    fields = cpc.cpc_subclass(duckdb.connect(), cfg).df()

    assert sorted(fields.field_id) == ["A61K", "H01L"]
