"""
Field definitions built from the CPC classification.

Three widths, because the width of a field mechanically changes everyone's
share and the advisor wants all three compared before one is chosen (Task 2):

    cpc_class       A61            broad
    cpc_subclass    A61K           middle
    cpc_main_group  A61K31/00      narrow

All three follow the contract in src/lib/fields.py and return a DuckDB relation
with columns (patent_id, field_id).

Two row-dropping rules live here rather than in the calling step, because both
are part of what "a field" *means* rather than sample restrictions:

  - the primary-code rule (`cpc.assignment`), which keeps only the examiner's
    main classification;
  - sample mode, which restricts everything to one CPC section for debugging.

Both go through the funnel when one is supplied, so they still appear in the
data diary. That is why these functions take a funnel argument at all.
"""

from src.lib import io


def _cpc_rows(con, cfg, funnel=None):
    """
    Read the CPC table and apply the rules shared by all three widths.

    Returns a relation with canonical column names — patent_id, cpc_section,
    cpc_class, cpc_subclass, cpc_group — whatever the source columns are called
    in this vintage of PatentsView.
    """
    # Which CPC table: as currently classified, or as classified at grant.
    # This is a look-ahead question, not a convenience one — see the warning in
    # config/default.yaml and notes/decisions.md C-3.
    table_key = f"cpc_{cfg.cpc.table}"
    path = io.interim_path(cfg, cfg.tables[table_key])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run: python src/01_convert_to_parquet.py"
        )

    columns = cfg.columns

    # This is the ONLY place where configured column names enter SQL. Everything
    # downstream uses the canonical names on the left of each AS, so the rest of
    # the queries in this file are literal SQL with nothing substituted into them.
    read_query = """
        SELECT
            "{patent_id}"   AS patent_id,
            "{sequence}"    AS cpc_sequence_text,
            "{section}"     AS cpc_section,
            "{cpc_class}"   AS cpc_class,
            "{subclass}"    AS cpc_subclass,
            "{group}"       AS cpc_group
        FROM read_parquet('{path}')
    """.format(
        patent_id=columns.patent_id,
        sequence=columns.cpc_sequence,
        section=columns.cpc_section,
        cpc_class=columns.cpc_class,
        subclass=columns.cpc_subclass,
        group=columns.cpc_group,
        path=path.as_posix(),
    )
    rows = con.sql(read_query)

    # Step 01 loads every column as text on purpose, so the sequence number has
    # to be converted here. A value that will not convert is a real problem with
    # the data, not something to quietly turn into a null, so it raises.
    unconvertible = rows.filter(
        "cpc_sequence_text IS NOT NULL AND TRY_CAST(cpc_sequence_text AS INTEGER) IS NULL"
    ).aggregate("count(*) AS n").fetchone()[0]
    if unconvertible:
        raise ValueError(
            f"{unconvertible:,} rows in {path.name} have a CPC sequence that is not a "
            f"whole number. Look at them before going further; do not cast them away."
        )
    rows = rows.project(
        "patent_id, CAST(cpc_sequence_text AS INTEGER) AS cpc_sequence, "
        "cpc_section, cpc_class, cpc_subclass, cpc_group"
    )

    # Sample mode: one CPC section only. Applied here so that everything
    # downstream inherits it through the patents that have a field at all.
    if cfg.sample_mode:
        rows = _drop(
            rows, funnel,
            name="sample_mode",
            condition=f"cpc_section = '{cfg.sample_cpc_section}'",
            why=(
                f"Sample mode is on, so the whole analysis is restricted to CPC section "
                f"{cfg.sample_cpc_section}. This is a debugging setting: it makes every step "
                f"run in minutes rather than hours, and it must be turned off in "
                f"config/default.yaml before any result is reported."
            ),
        )

    # Most patents carry several CPC codes. Keeping only the primary one fixes,
    # once and for all, which field a patent belongs to.
    if cfg.cpc.assignment == "primary":
        before_primary = rows
        rows = _drop(
            rows, funnel,
            name="cpc_primary_only",
            condition=f"cpc_sequence = {cfg.cpc.primary_sequence}",
            why=(
                f"Each patent is assigned to its primary CPC code only (sequence "
                f"{cfg.cpc.primary_sequence}), as the advisor's Task 1 requires. Most patents "
                f"carry several codes, and counting a patent in every field it touches would "
                f"inflate field sizes and dilute every inventor's share. The all-codes "
                f"alternative is available by setting cpc.assignment to 'all'."
            ),
        )
        _check_primary_sequence_matched(before_primary, rows, cfg)

    return rows


def _check_primary_sequence_matched(before, after, cfg):
    """
    A primary-code rule that matches nothing is a configuration error, and it
    has to be caught here rather than five joins later.

    `cpc.primary_sequence` belongs to the table named in `cpc.table`, not to CPC
    itself: g_cpc_current numbers its codes from 0 and g_cpc_at_issue from 1. So
    changing `cpc.table` alone leaves this filter matching no rows, every patent
    loses its field, and the run does not fail at the setting that caused it —
    it fails much later with an empty universe, which reads like a data problem.
    Measured at Task 2b item 5; see notes/decisions.md C-33.

    LIMIT 1 rather than a count: the question is whether ANY row survives, and
    on a 59-million-row table the difference is a second against a scan.
    """
    if after.limit(1).fetchone() is not None:
        return

    lowest = before.aggregate("min(cpc_sequence) AS lowest").fetchone()[0]
    if lowest is None:
        raise ValueError(
            f"No CPC rows at all before the primary-code rule was applied. Either "
            f"{cfg.tables['cpc_' + cfg.cpc.table]} is empty, or sample mode "
            f"({cfg.sample_cpc_section}) selected a section that is not in it.")
    raise ValueError(
        f"cpc.primary_sequence = {cfg.cpc.primary_sequence} matches no rows in "
        f"{cfg.tables['cpc_' + cfg.cpc.table]}, whose lowest sequence is {lowest}. "
        f"These two settings are not independent: the sequence numbering belongs to "
        f"the table. Set cpc.primary_sequence to {lowest} to keep the primary-code "
        f"rule, or cpc.assignment to 'all' to drop it. Continuing would give every "
        f"patent no field and empty the whole analysis.")


def _add_field_id(rows, expression, funnel, strategy_name):
    """Attach the field_id, then drop patents that could not be placed in a field."""
    with_field = rows.project(f"patent_id, {expression} AS field_id")
    return _drop(
        with_field, funnel,
        name=f"field_assigned_{strategy_name}",
        condition="field_id IS NOT NULL AND field_id <> ''",
        why=(
            f"A patent with no usable CPC code at the {strategy_name} level cannot be placed "
            f"in a field, and a patent outside every field cannot contribute to anyone's "
            f"share of one. These rows are dropped rather than pooled into a residual "
            f"category, which would behave like an enormous artificial field."
        ),
    ).project("patent_id, field_id")


def _drop(relation, funnel, name, condition, why):
    """Filter through the funnel when there is one; plain filter when there is not."""
    if funnel is None:
        return relation.filter(condition)
    return funnel.filter(relation, name, condition, why)


# ---------------------------------------------------------------------------
# The three strategies
# ---------------------------------------------------------------------------

def cpc_class(con, cfg, funnel=None):
    """Broad fields: the three-character CPC class, e.g. A61 (medical and veterinary)."""
    rows = _cpc_rows(con, cfg, funnel=funnel)
    return _add_field_id(rows, "cpc_class", funnel, "cpc_class")


def cpc_subclass(con, cfg, funnel=None):
    """Middle fields: the four-character CPC subclass, e.g. A61K (medical preparations)."""
    rows = _cpc_rows(con, cfg, funnel=funnel)
    return _add_field_id(rows, "cpc_subclass", funnel, "cpc_subclass")


def cpc_main_group(con, cfg, funnel=None):
    """
    Narrow fields: the CPC main group, e.g. A61K31/00.

    PatentsView stores the full subgroup (A61K31/4402). The main group is
    everything before the slash, with /00 restored: A61K31/00. A code with no
    slash at all is left as it is and given the same suffix, which keeps it in
    its own field rather than silently merging it with another.
    """
    rows = _cpc_rows(con, cfg, funnel=funnel)
    expression = "split_part(cpc_group, '/', 1) || '/00'"
    return _add_field_id(rows, expression, funnel, "cpc_main_group")
