"""
The registry of field definitions.

Why this file exists
--------------------
"Field" is the most consequential researcher choice in this design. A star's
share of a field mechanically depends on how wide the field is drawn: 4% of a
CPC class and 4% of a CPC main group are entirely different claims. The research
design commits to reporting that sensitivity as a *result*, not a footnote, and
to never choosing the delineation after seeing the estimates.

That commitment only survives if switching definitions is trivial. So a field
definition is a named function, and the config picks one by name. Nothing else
in the code knows which definition is in use.

The contract
------------
A field strategy is a function:

    strategy(con, cfg, funnel=None) -> DuckDB relation with columns (patent_id, field_id)

  con     an open DuckDB connection
  cfg     the loaded config, so the strategy can find its own inputs
  funnel  optional; if given, any row-dropping the strategy does is recorded

That is deliberately general. The CPC strategies read the CPC table and cut the
code at different depths. A future embedding strategy will read a table of
patent vectors and return neighbourhoods around each star's pre-death corpus —
a completely different computation, but the same two output columns, so no
caller has to change.

Adding a definition later
-------------------------
1. Write src/lib/field_strategies/<something>.py with a function following the
   contract above.
2. Add one line to FIELD_STRATEGIES below.

That is all. Config validation reads this dictionary, so the new name becomes a
legal value of `fields.strategy` immediately; no caller, and no validator, has
to be edited. Two definitions are planned and NOT authorised yet:
embedding_neighborhood (PatentSBERTa) and pharma_target_class.
"""

from src.lib.field_strategies import cpc


# A plain dictionary of functions. This is the one place where the repository
# uses indirection on purpose, because which field definition is right is a
# reported result and not a fixed choice — and it stays a plain dictionary.
FIELD_STRATEGIES = {
    "cpc_class": cpc.cpc_class,            # broad,  e.g. A61
    "cpc_subclass": cpc.cpc_subclass,      # middle, e.g. A61K
    "cpc_main_group": cpc.cpc_main_group,  # narrow, e.g. A61K31/00
}


def available_strategies():
    """The names that are legal values of `fields.strategy`. Used by config validation."""
    return sorted(FIELD_STRATEGIES)


def assign_fields(con, cfg, funnel=None, strategy=None):
    """
    Map every patent to a field, using the strategy named in the config.

    Returns a DuckDB relation with two columns: patent_id, field_id. A patent
    appears more than once if `cpc.assignment` is "all" and it carries several
    codes; exactly once if the assignment is "primary".

    `strategy` overrides `fields.strategy` for one call. It exists for Task 2,
    which builds all three widths in a single run to compare them: the
    alternative would be mutating the config between widths, and a config that
    changes while a run is in progress is exactly the thing the hash in every
    output filename is meant to rule out.
    """
    name = strategy or cfg.fields.strategy
    if name not in FIELD_STRATEGIES:
        raise KeyError(
            f"'{name}' is not a registered field definition. "
            f"Available: {', '.join(available_strategies())}."
        )
    return FIELD_STRATEGIES[name](con, cfg, funnel=funnel)
