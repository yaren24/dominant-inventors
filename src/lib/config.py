"""
Loads, validates and fingerprints config/default.yaml.

Why this file exists
--------------------
Every analytical choice in the thesis lives in a YAML file, so the YAML file is
the most dangerous file in the repository: a typo in a key name would otherwise
be silently ignored, and the analysis would quietly run on the default instead
of the value you thought you set. That failure produces plausible numbers and no
error message, which is the worst kind.

So nothing here is forgiving. Loading a config either returns a fully checked
set of values or raises with a list of everything that is wrong. Three things
are checked:

  unknown key      you wrote `margin_cpa` and meant `margin_cap`
  missing key      the code needs a value the file does not give
  bad value        a share above 1, a negative window, "fractoinal"

There is also a fourth check that is easy to miss and matters more than the
other three: keys that must agree with each other. The 1986 death floor is not
a free choice — it is 1976 (data start) + 5 (to build the first dominance
window) + 5 (pre-death years). If someone shortens the window for a robustness
check and leaves 1986 in place, the pre-period is silently half empty. So the
code recomputes it and refuses to run if the two disagree.

Usage
-----
    from src.lib import config

    cfg = config.load("config/default.yaml")
    cfg.dominance.margin_cap      # 10
    cfg.windows.pre_years         # 5
    cfg.hash                      # 'a3f19c02' — stamps every output file
"""

import difflib
import hashlib
import json
from pathlib import Path

import yaml

from src.lib import fields


# ---------------------------------------------------------------------------
# The schema: every key the config is allowed to contain.
# ---------------------------------------------------------------------------
# One line per key. The *explanation* of what a key means lives in the comments
# in config/default.yaml, where you read it; this table only says what a legal
# value looks like, so that an illegal one is caught before it reaches a result.
#
#   type          str, int, float, bool, list  (int is accepted where float is asked)
#   choices       the complete list of allowed values
#   min / max     inclusive bounds
#   allow_none    the value may be null in the YAML (meaning "not decided yet")
#   item_*        the same checks applied to every element of a list
#
# Adding a key means adding a line here AND a line in config/default.yaml. The
# test suite fails if the two ever drift apart.

YEAR = {"type": int, "min": 1900, "max": 2100}
SHARE = {"type": float, "min": 0.0, "max": 1.0}
POSITIVE_YEARS = {"type": int, "min": 1, "max": 100}
NAME = {"type": str}

SCHEMA = {
    "config_name": NAME,
    "random_seed": {"type": int, "min": 0},

    "sample_mode": {"type": bool},
    "sample_cpc_section": {"type": str, "choices": list("ABCDEFGHY")},

    "paths.raw_patentsview": NAME,
    "paths.raw_kjl": NAME,
    "paths.interim": NAME,
    # Where the OTHER admission arm's interim files live, for Task 3c's
    # side-by-side comparison. A path and nothing else: it names no analytical
    # choice, which is why it sits here and not in a task3c block, and why the
    # config hash does not move when it changes (paths. is in NOT_IN_HASH).
    "paths.comparison_interim": NAME,
    # The config files the floor-and-margin comparison step reads side by side.
    # Paths and nothing else, like comparison_interim: naming which runs a table
    # compares is not an analytical choice, so it lives under paths. and stays
    # out of the hash.
    "paths.floor_robustness_configs": {"type": list, "item_type": str},
    "paths.processed": NAME,
    "paths.outputs": NAME,
    "paths.logs": NAME,

    "tables.patents": NAME,
    "tables.application": NAME,
    "tables.inventors": NAME,
    "tables.assignees": NAME,
    "tables.cpc_current": NAME,
    "tables.cpc_at_issue": NAME,
    "tables.citations": NAME,
    "tables.locations": NAME,
    "tables.persistent_inventor": NAME,
    "tables.kjl_ages": NAME,
    "tables.kjl_deaths": NAME,
    "tables.kjl_crosswalk": NAME,

    "columns.patent_id": NAME,
    "columns.patent_type": NAME,
    "columns.patent_date": NAME,
    "columns.withdrawn": NAME,
    "columns.filing_date": NAME,
    "columns.inventor_id": NAME,
    "columns.inventor_sequence": NAME,
    "columns.assignee_id": NAME,
    "columns.assignee_sequence": NAME,
    "columns.location_id": NAME,
    "columns.cpc_sequence": NAME,
    "columns.cpc_section": NAME,
    "columns.cpc_class": NAME,
    "columns.cpc_subclass": NAME,
    "columns.cpc_group": NAME,
    "columns.cpc_type": NAME,
    "columns.cpc_version": NAME,
    "columns.citation_patent_id": NAME,
    "columns.citation_date": NAME,
    "columns.citation_category": NAME,
    "columns.persistent_kjl_vintage": NAME,
    "columns.persistent_current_vintage": NAME,
    "columns.kjl_inventor_id": NAME,
    "columns.kjl_birth_year": NAME,
    "columns.kjl_death_birth_year": NAME,
    "columns.kjl_death_date": NAME,
    "columns.kjl_death_year": NAME,
    "columns.kjl_death_source": NAME,
    "columns.kjl_death_score": NAME,

    "patent_universe.date_by": {"type": str, "choices": ["application", "grant"]},
    "patent_universe.patent_types": {
        "type": list,
        "item_type": str,
        "item_choices": ["utility", "design", "plant", "reissue", "defensive publication",
                         "statutory invention registration"],
    },
    "patent_universe.first_year": YEAR,
    "patent_universe.filings_complete_through": YEAR,
    "patent_universe.drop_withdrawn": {"type": bool},
    "patent_universe.drop_incomplete_filing_years": {"type": bool},
    "patent_universe.inventor_country": {"type": str, "allow_none": True},

    "citations.deduplicate_edges": {"type": bool},
    "citations.category_conflict_rule": {
        "type": str,
        "choices": ["any_examiner", "all_examiner", "drop_conflicts"],
    },

    "counting.method": {"type": str, "choices": ["fractional", "full"]},
    "counting.field_size_method": {"type": str, "choices": ["fractional", "full"]},

    "cpc.assignment": {"type": str, "choices": ["primary", "all"]},
    "cpc.primary_sequence": {"type": int, "min": 0},
    "cpc.table": {"type": str, "choices": ["current", "at_issue"]},
    "cpc.multi_code_weighting": {"type": str, "choices": ["duplicate", "fractional"]},

    # NOTE: fields.strategy is checked against the registry in src/lib/fields.py,
    # not against a list written here. That is what lets a new strategy file
    # become a legal config value without anyone editing this module.
    "fields.strategy": {"type": str},
    "fields.robustness_strategies": {"type": list, "item_type": str},

    "windows.dominance_window_years": POSITIVE_YEARS,
    "windows.pre_years": POSITIVE_YEARS,
    "windows.post_years": POSITIVE_YEARS,

    "deaths.year_min": YEAR,
    "deaths.year_max": YEAR,
    "deaths.check_window_consistency": {"type": bool},
    "deaths.age_at_death_max": {"type": int, "min": 0, "max": 120},
    "deaths.age_at_death_inclusive": {"type": bool},
    "deaths.age_at_death_robustness": {"type": list, "item_type": int, "item_min": 0,
                                       "item_max": 120},
    "deaths.actively_inventing_required": {"type": bool},
    "deaths.actively_inventing_within_years": POSITIVE_YEARS,
    "deaths.actively_inventing_robustness_years": {"type": list, "item_type": int,
                                                   "item_min": 1, "item_max": 100},

    # There is no general_pool_rule to match treated_pool_rule. Rule B needs a
    # death record, so it cannot assign a birth year to a living inventor and
    # cannot define a general pool — a key whose only legal value is "loose"
    # would be a decision that looks like a choice.
    "birth_years.eligibility_min_score": {"type": float},
    "birth_years.eligibility_score_inclusive": {"type": bool},
    "birth_years.rule_b_tolerance_years": {"type": int, "min": 0, "max": 20},
    "birth_years.treated_pool_rule": {"type": str, "choices": ["strict", "loose"]},

    "deaths.min_source_score": {"type": float, "min": 0},
    "deaths.tie_rule": {"type": str,
                        "choices": ["modal_then_earliest", "earliest", "latest", "drop"]},

    "kjl_link.id_suffix_base": {"type": int, "min": 0, "max": 1},
    "kjl_link.require_vintage_agreement": {"type": bool},
    "kjl_link.repair_sir_patent_ids": {"type": bool},

    "dominance.metric": {"type": str, "choices": ["patent_share", "citation_share"]},
    "dominance.first_compute_year": YEAR,
    "dominance.last_compute_year": YEAR,
    "dominance.rank_tie_rule": {"type": str,
                                "choices": ["competition", "dense", "unique_by_id"]},
    "dominance.weight_rounding_decimals": {"type": int, "min": 0, "max": 15},
    "dominance.margin_when_no_runner_up": {"type": str, "choices": ["missing", "cap"]},
    "dominance.compute_citation_share": {"type": bool},
    "dominance.citation_share.count_citations_through": {"type": str,
                                                         "choices": ["as_of", "all_time"]},
    "dominance.citation_share.exclude_self_citations": {"type": bool},
    "dominance.cutoff": SHARE,
    "dominance.cutoffs": {"type": list, "item_type": float, "item_min": 0.0, "item_max": 1.0},
    # Task 3b's two grids. Separate keys rather than one shared list, because a
    # count-share cutoff and a knowledge-base-share cutoff are different objects
    # that agree numerically today; see config/default.yaml for the argument.
    "dominance.reach_cutoffs": {"type": list, "item_type": float, "item_min": 0.0, "item_max": 1.0},
    "dominance.kb_share_cutoffs": {"type": list, "item_type": float, "item_min": 0.0,
                                   "item_max": 1.0},
    "dominance.margin_cap": {"type": float, "min": 1.0},
    "dominance.alternative_rule.max_rank": {"type": int, "min": 1},
    "dominance.alternative_rule.min_margin": {"type": float, "min": 1.0},
    "dominance.alternative_rule.min_share": SHARE,
    "dominance.min_field_patents": {"type": int, "min": 0},
    "dominance.min_field_patents_inclusive": {"type": bool},
    # The floor as a percentile of the field-year window-size distribution rather
    # than an absolute count (advisor, 2026-09-15: "100 is an arbitrary number").
    # `floor_percentile` and `floor_scope` mean something only under `percentile`
    # and must be null under `absolute`; _check_consistency enforces both
    # directions, so no value here is ever validated and then read by nothing.
    "dominance.floor_mode": {"type": str, "choices": ["absolute", "percentile"]},
    "dominance.floor_percentile": {"type": float, "min": 0.0, "max": 100.0, "allow_none": True},
    "dominance.floor_scope": {"type": str, "choices": ["pooled", "per_year"], "allow_none": True},
    "dominance.apply_min_field_size": {"type": bool},
    "dominance.report_both_min_field_size": {"type": bool},

    # The percentiles the floor diagnostic tabulates as candidate floors (advisor,
    # 2026-09-15: "the absolute floor implied by the 10th, 25th, 50th and 75th
    # percentiles"). Presentation of a distribution, read by step 02d only.
    "floor_diagnostic.percentiles": {"type": list, "item_type": float, "item_min": 0.0,
                                     "item_max": 100.0},

    "task2.widths": {"type": list, "item_type": str},
    "task2.histogram_bins": {"type": int, "min": 5, "max": 1000},
    "task2.field_size_groups": {"type": str, "choices": ["quartiles", "fixed_bins"]},
    "task2.field_size_fixed_bins": {"type": list, "item_type": int, "item_min": 1},

    "task2b.item1_top_n": {"type": int, "min": 1},
    "task2b.item1_share_bands": {"type": list, "item_type": float,
                                 "item_min": 0.0, "item_max": 1.0},
    "task2b.item2_top_n": {"type": int, "min": 1},
    # A share of a code's at-issue rows, so between 0 and 1. 0 is legal and
    # means "take the plain minimum", which is the comparison item 6 reports
    # against; 1 would require every row to agree and is legal for the same
    # reason — both ends of the ladder have to be reachable from the config.
    "task2b.item6_version_min_share": {"type": float, "min": 0.0, "max": 1.0},

    "task4.old_cohort_max_birth_year": YEAR,
    "task4.birth_cohort_bin_years": {"type": int, "min": 1, "max": 50},
    "task4.suspicious_career_years": POSITIVE_YEARS,
    # How many measurement windows the loose timing rule reaches back through.
    # 1 collapses it onto the strict rule and is legal on purpose: both ends of
    # the ladder have to be reachable from the config.
    "task4.loose_timing_span_years": POSITIVE_YEARS,
    "task4.undatable_age_reporting": {
        "type": str,
        "choices": ["counted_with_column", "counted_silently", "excluded"]},
    "task4.dose_percentile_population": {
        "type": str, "choices": ["all_rows", "floored_fields"]},
    "task4.type_split_grid": {
        "type": str, "choices": ["every_cutoff_pair", "working_pair"]},
    # Two whole years, low then high. The pair is validated as a list of ints
    # rather than two keys because a range whose ends can be set independently
    # can be set the wrong way round; check_consistency below pins the order.
    "task4.age_at_death_plausible_range": {"type": list, "item_type": int,
                                           "item_min": 0, "item_max": 200},

    # Yaren's overlap-and-nesting brief of 2026-09-19. The era starts, the lag
    # cutoffs and the first era's position are pinned to other keys in
    # _check_consistency below; here each value is checked on its own.
    "task_overlap.era_start_years": {"type": list, "item_type": int,
                                     "item_min": 1900, "item_max": 2100},
    "task_overlap.lag_years": POSITIVE_YEARS,
    "task_overlap.lag_share_cutoffs": {"type": list, "item_type": float,
                                       "item_min": 0.0, "item_max": 1.0},
    "task_overlap.lag_reach_cutoffs": {"type": list, "item_type": float,
                                       "item_min": 0.0, "item_max": 1.0},
    "task_overlap.lag_kb_share_cutoffs": {"type": list, "item_type": float,
                                          "item_min": 0.0, "item_max": 1.0},
    "task_overlap.matched_top_fractions": {"type": list, "item_type": float,
                                           "item_min": 0.0, "item_max": 1.0},
    "task_overlap.distribution_percentiles": {"type": list, "item_type": float,
                                              "item_min": 0.0, "item_max": 1.0},
    "task_overlap.top_ranks": {"type": list, "item_type": int, "item_min": 1},

    "task5.employer_window_years": POSITIVE_YEARS,
    "task5.multi_death_field_window_years": POSITIVE_YEARS,

    "task6.living_sample_size": {"type": int, "min": 1},
    "task6.reference_year_min": YEAR,
    "task6.reference_year_max": YEAR,

    "task7.match_same_cpc_section": {"type": bool},
    "task7.field_size_ratio_max": {"type": float, "min": 1.0},
    "task7.dominance_ratio_max": {"type": float, "min": 1.0},
    "task7.no_death_within_years": POSITIVE_YEARS,
    "task7.match_on_hhi": {"type": bool},
    "task7.match_on_pretrend": {"type": bool},
    "task7.calendar_year_tolerance": {"type": int, "min": 0, "allow_none": True},
    "task7.purge_cpc_overlapping_controls": {"type": bool},

    "task8.n_events_grid": {"type": list, "item_type": int, "item_min": 1},
    "task8.n_replications": {"type": int, "min": 1},
    "task8.mde_multiplier": {"type": float, "min": 0.0},
    "task8.reference_effects": {"type": list, "item_type": float, "item_min": 0.0,
                                "item_max": 10.0},

    "task9.rival_active_window_years": POSITIVE_YEARS,
    "task9.rival_min_patents": {"type": int, "min": 1},

    "estimation.cluster_level": {"type": str, "choices": ["death", "inventor_field", "field"]},
    "estimation.stacked": {"type": bool},

    "subsample.which": {"type": str, "choices": ["all", "pharma", "semiconductors"]},
    "subsample.pharma_cpc": {"type": list, "item_type": str},
    "subsample.semiconductor_cpc": {"type": list, "item_type": str},

    "birth_years.gap_table_top_code": {"type": int, "min": 1, "max": 30},

    "task2b.item1_field_rows": {"type": int, "min": 1},
    "task2b.item6_field_rows": {"type": int, "min": 1},
    "task2b.item9_materiality": {"type": float, "min": 0.0, "max": 1.0},

    # ---- Task 2c, the citation family. C-48 to C-52, all PROVISIONAL. ----
    "task2c.widths": {"type": list, "item_type": str},
    "task2c.first_compute_year": {"type": int, "min": 1976, "max": 2022},
    "task2c.store_objects": {"type": list, "item_type": str},
    "task2c.apply_cutoff": {"type": bool},
    "task2c.normalisation": {
        "type": str,
        "choices": ["none", "ratio_to_field_year", "within_year_percentile"],
    },
    "task2c.cohort_bin_years": {"type": int, "min": 1, "max": 31},
    "task2c.primary_measure": {"type": str, "choices": ["kb_share", "reach"]},
    "task2c.kill_condition_measure": {
        "type": str,
        "choices": ["kb_share_fractional", "kb_share", "reach"],
    },
    "task2c.margin_denominator": {"type": str, "choices": ["common", "per_inventor"]},
    "task2c.kb_denominator": {"type": str, "choices": ["in_universe_1976plus"]},
    "task2c.flag_corpus_left_censored_before": {"type": int, "min": 1976, "max": 2000},
    "task2c.citing_admission": {"type": str, "choices": ["filing", "grant_strict"]},
    "task2c.cited_must_be_granted_by_t": {"type": bool},
    "task2c.dose_lag_years": {"type": int, "min": 0, "max": 5},
    # `noisy_or` was removed as a choice, not merely defaulted away from. C-52
    # D107 had already rejected it on economics — it is not a crediting rule but
    # an independence assumption, and co-invention is the least independent event
    # in this data — and it was also WRONG in arithmetic: DuckDB's sum() skips
    # NULLs rather than propagating them, so the nullif() guarding ln(0) made a
    # SOLO cited patent contribute nothing instead of forcing credit to 1. A star
    # cited only through her solo work scored NULL, the strongest possible
    # dependence link read as no link, and no guard fired because a NULL is not
    # outside [0, 1]. D107 measured 75-92% of (citing patent, cited inventor)
    # pairs citing exactly one of the star's patents and the median star 48.8%
    # solo, so it was wrong on most of the data. A legal setting that yields a
    # complete, plausible, silently wrong panel is the one thing a choices list
    # must not contain. See C-55.
    "task2c.reach_fractional_rule": {
        "type": str,
        "choices": ["max_credit", "mean", "sum"],
    },
    "task2c.min_usable_reference_share": {"type": float, "min": 0.0, "max": 1.0},

    "checkpoints.share_decimals": {"type": int, "min": 0, "max": 12},
    "checkpoints.treated_comfortable": {"type": int, "min": 1},
    "checkpoints.treated_workable_min": {"type": int, "min": 1},
    "checkpoints.correlation_collapse": {"type": float, "min": 0.0, "max": 1.0},
    "checkpoints.task7_min_candidates": {"type": int, "min": 1},
    "checkpoints.task8_mde_concern": {"type": float, "min": 0.0},

    "runtime.duckdb_memory_limit": NAME,
    "runtime.duckdb_temp_directory": NAME,
    "runtime.duckdb_threads": {"type": int, "min": 1, "allow_none": True},
}


# Keys deliberately left out of the config hash, because changing them does not
# change a single number: the name of the output folder, where files live, and
# how much memory DuckDB is allowed. Everything else is in, including
# sample_mode — running on one CPC section gives different results, so it must
# give a different hash.
#
# `runtime.` is here for a specific reason: giving DuckDB more memory would
# otherwise change the hash and therefore every output filename, which would
# make two identical analyses look like two different results.
NOT_IN_HASH = ("config_name", "paths.", "runtime.")


class ConfigError(Exception):
    """Raised when a config file is wrong. The message lists everything at once."""


# ---------------------------------------------------------------------------
# Reading values with dots: cfg.dominance.margin_cap
# ---------------------------------------------------------------------------

class ConfigSection:
    """
    A nested dictionary you read with dots instead of brackets.

    This is a class rather than a plain dict for one reason: when you mistype a
    key name, a dict gives you a bare KeyError, whereas this tells you which
    section you were in and what the nearby names are.
    """

    def __init__(self, values, prefix=""):
        self._prefix = prefix
        self._values = {}
        for key, value in values.items():
            if isinstance(value, dict):
                self._values[key] = ConfigSection(value, f"{prefix}{key}.")
            else:
                self._values[key] = value

    def __getattr__(self, name):
        # Python calls this only when normal attribute lookup has already failed.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            near = difflib.get_close_matches(name, self._values.keys(), n=3)
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise AttributeError(
                f"There is no config key '{self._prefix}{name}'.{hint}\n"
                f"Available in this section: {', '.join(sorted(self._values))}"
            ) from None

    def __getitem__(self, name):
        """cfg.tables["cpc_current"], for when the key name is itself computed."""
        return self.__getattr__(name)

    def to_dict(self):
        """Plain nested dictionaries again — for writing the config back out."""
        return {
            key: value.to_dict() if isinstance(value, ConfigSection) else value
            for key, value in self._values.items()
        }

    def __repr__(self):
        return f"<config section {self._prefix or 'root'}: {', '.join(sorted(self._values))}>"


class Config(ConfigSection):
    """A whole validated config file, plus its fingerprint."""

    def __init__(self, values, source_path, source_text):
        super().__init__(values)
        self._source_path = Path(source_path)
        self._source_text = source_text
        self._hash = _config_hash(values)

    @property
    def hash(self):
        """8 hex characters that identify this exact set of analytical choices."""
        return self._hash

    @property
    def name(self):
        """The config name, which names the folder under outputs/."""
        return self._values["config_name"]

    @property
    def source_path(self):
        return self._source_path

    @property
    def source_text(self):
        """The YAML file exactly as written, for the verbatim run snapshot."""
        return self._source_text

    def __repr__(self):
        return f"<Config '{self.name}' hash={self.hash} from {self._source_path}>"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(path="config/default.yaml"):
    """Read a config file, check it completely, and return it. Raises on any problem."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"No config file at {path}. Expected e.g. config/default.yaml.")

    source_text = path.read_text(encoding="utf-8")
    values = yaml.safe_load(source_text)
    if not isinstance(values, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping of keys to values.")

    problems = _find_problems(values)
    if problems:
        listed = "\n".join(f"  {i}. {p}" for i, p in enumerate(problems, start=1))
        raise ConfigError(
            f"{len(problems)} problem(s) in {path}:\n{listed}\n\n"
            f"Nothing was run. Fix the file and try again."
        )

    return Config(values, path, source_text)


def flatten(cfg):
    """
    The whole config as one flat dictionary: {'dominance.margin_cap': 10, ...}.

    Used to compare one run against the previous one, so that LATEST.md can say
    exactly which settings changed.
    """
    return _flatten(cfg.to_dict())


def _find_problems(values):
    """Return a list of everything wrong with this config, in plain English."""
    flat = _flatten(values)
    problems = []

    # 1. Keys that are in the file but not in the schema — usually a typo.
    for key in sorted(set(flat) - set(SCHEMA)):
        near = difflib.get_close_matches(key, SCHEMA.keys(), n=2)
        hint = f" Did you mean '{near[0]}'?" if near else ""
        problems.append(f"Unknown key '{key}'.{hint}")

    # 2. Keys the code needs that the file does not have.
    for key in sorted(set(SCHEMA) - set(flat)):
        problems.append(f"Missing key '{key}'. Every key in the schema must be present.")

    # 3. Values that are the wrong type or out of range.
    for key in sorted(set(flat) & set(SCHEMA)):
        problem = _check_value(key, flat[key], SCHEMA[key])
        if problem:
            problems.append(problem)

    # 4. Keys that have to agree with each other. Only worth checking if the
    #    individual values are sane, hence the guard.
    if not problems:
        problems.extend(_check_consistency(values))

    return problems


def _flatten(values, prefix=""):
    """{'a': {'b': 1}} becomes {'a.b': 1}. Lists are values, not sections."""
    flat = {}
    for key, value in values.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _check_value(key, value, rule):
    """Check one value against one schema rule. Returns a message, or None if fine."""
    if value is None:
        if rule.get("allow_none"):
            return None
        return f"'{key}' is empty. It needs a value."

    expected = rule["type"]

    # bool is a subclass of int in Python, so it has to be checked first or
    # `true` would silently pass as an integer.
    if expected is bool:
        if not isinstance(value, bool):
            return f"'{key}' should be true or false, not {value!r}."
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"'{key}' should be a number, not {value!r}."
    elif expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"'{key}' should be a whole number, not {value!r}."
    elif not isinstance(value, expected):
        return f"'{key}' should be {expected.__name__}, not {type(value).__name__} ({value!r})."

    if "choices" in rule and value not in rule["choices"]:
        return f"'{key}' is {value!r}, but the only allowed values are: {rule['choices']}."

    if "min" in rule and value < rule["min"]:
        return f"'{key}' is {value}, which is below the minimum of {rule['min']}."
    if "max" in rule and value > rule["max"]:
        return f"'{key}' is {value}, which is above the maximum of {rule['max']}."

    if expected is list:
        for i, item in enumerate(value):
            item_rule = {
                k[len("item_"):]: v for k, v in rule.items() if k.startswith("item_")
            }
            if item_rule:
                problem = _check_value(f"{key}[{i}]", item, item_rule)
                if problem:
                    return problem

    return None


def _check_consistency(values):
    """
    Checks between keys, where each value is fine on its own but the combination
    is not. These are the ones that would otherwise produce a plausible, wrong
    number instead of an error.
    """
    problems = []
    universe = values["patent_universe"]
    windows = values["windows"]
    deaths = values["deaths"]

    # The gap table's "N or more" row must sit STRICTLY ABOVE the tolerance it
    # exists to justify. At any tolerance >= the top code, the row that decides
    # pass from fail fuses into the top-coded bucket, inside the very table
    # whose adjacent prose calls itself the argument for that tolerance. Both
    # 00b and 01 print that table, so the check lives here and fires for every
    # step at load rather than being duplicated in two of them.        [C-44]
    births = values["birth_years"]
    if births["gap_table_top_code"] <= births["rule_b_tolerance_years"]:
        problems.append(
            f"birth_years.gap_table_top_code ({births['gap_table_top_code']}) must be "
            f"strictly greater than birth_years.rule_b_tolerance_years "
            f"({births['rule_b_tolerance_years']}). At this combination the row that "
            f"decides Rule B pass from fail fuses into the "
            f"'{births['gap_table_top_code']} or more' bucket, in the table whose own "
            f"prose is the argument for the tolerance. Raise the top code above the "
            f"tolerance, or lower the tolerance."
        )

    # A range whose two ends can be set independently can be set the wrong way
    # round, and `low > high` makes every age implausible — which reads as "the
    # whole treated sample has an unusable age" rather than as a config error.
    # Exactly two values, because a third would be silently ignored.  [C-99]
    span = values["task4"]["age_at_death_plausible_range"]
    if len(span) != 2 or span[0] >= span[1]:
        problems.append(
            f"task4.age_at_death_plausible_range is {span}. It must be exactly two years, "
            f"low then high, with the low strictly below the high. Reversed, every age at "
            f"death is flagged implausible and the note reads as a finding about the death "
            f"data rather than as a mis-typed config."
        )

    if universe["first_year"] > universe["filings_complete_through"]:
        problems.append(
            f"patent_universe.first_year ({universe['first_year']}) is after "
            f"filings_complete_through ({universe['filings_complete_through']})."
        )

    if deaths["year_min"] > deaths["year_max"]:
        problems.append(
            f"deaths.year_min ({deaths['year_min']}) is after deaths.year_max "
            f"({deaths['year_max']})."
        )

    # The death window is implied by the data start and the two window lengths.
    # See notes/decisions.md, A-2: this is the check that stops a shortened
    # window from silently leaving the pre-death period half empty.
    if deaths["check_window_consistency"]:
        earliest = (universe["first_year"]
                    + windows["dominance_window_years"]
                    + windows["pre_years"])
        if deaths["year_min"] < earliest:
            problems.append(
                f"deaths.year_min is {deaths['year_min']}, but with data starting in "
                f"{universe['first_year']}, a {windows['dominance_window_years']}-year "
                f"dominance window and {windows['pre_years']} pre-death years, the "
                f"earliest usable death year is {earliest}. Either raise "
                f"deaths.year_min to {earliest} or shorten the windows."
            )
        latest = universe["filings_complete_through"] - windows["post_years"]
        if deaths["year_max"] > latest:
            problems.append(
                f"deaths.year_max is {deaths['year_max']}, but with filings complete only "
                f"through {universe['filings_complete_through']} and {windows['post_years']} "
                f"post-death years needed, the latest usable death year is {latest}. Either "
                f"lower deaths.year_max to {latest} or shorten the post-death window."
            )

    # Field strategies are checked against the registry itself, so that adding a
    # strategy file is enough to make its name legal here.
    available = fields.available_strategies()
    if values["fields"]["strategy"] not in available:
        problems.append(
            f"fields.strategy is '{values['fields']['strategy']}', which is not a "
            f"registered field definition. Available: {', '.join(available)}. "
            f"To add one, see docs/how_to_change_things.md."
        )
    for name in values["fields"]["robustness_strategies"]:
        if name not in available:
            problems.append(
                f"fields.robustness_strategies contains '{name}', which is not a "
                f"registered field definition. Available: {', '.join(available)}."
            )
    for name in values["task2"]["widths"]:
        if name not in available:
            problems.append(
                f"task2.widths contains '{name}', which is not a registered field "
                f"definition. Available: {', '.join(available)}."
            )

    # The years dominance is measured for have to lie inside the years patents
    # exist for, or a window would be built from data that is not there.
    dominance = values["dominance"]
    if dominance["first_compute_year"] > dominance["last_compute_year"]:
        problems.append(
            f"dominance.first_compute_year ({dominance['first_compute_year']}) is after "
            f"dominance.last_compute_year ({dominance['last_compute_year']})."
        )
    if dominance["first_compute_year"] < universe["first_year"]:
        problems.append(
            f"dominance.first_compute_year is {dominance['first_compute_year']}, before "
            f"patent data begins in {universe['first_year']}."
        )
    if dominance["last_compute_year"] > universe["filings_complete_through"]:
        problems.append(
            f"dominance.last_compute_year is {dominance['last_compute_year']}, but filings "
            f"are only complete through {universe['filings_complete_through']}. A window "
            f"closing later would be measured on a year the data has not finished."
        )
    # Computing partial windows is a legitimate choice, but it has to be a
    # deliberate one, because a one-year window and a five-year window produce
    # shares that are not comparable. Same switch as the death-window check
    # above: this is a guard against forgetting, not a prohibition.
    if deaths["check_window_consistency"]:
        first_full = universe["first_year"] + windows["dominance_window_years"] - 1
        if dominance["first_compute_year"] < first_full:
            problems.append(
                f"dominance.first_compute_year is {dominance['first_compute_year']}, but the "
                f"first year with a COMPLETE {windows['dominance_window_years']}-year window "
                f"is {first_full}. Earlier years would get a shorter window and a share that "
                f"is not comparable with the rest. Set it to {first_full}, or set "
                f"deaths.check_window_consistency to false if the partial windows are wanted "
                f"on purpose."
            )

    # The field floor: under `absolute` the percentile settings must be empty,
    # because nothing reads them and a number left there would move the hash
    # without moving a floor; under `percentile` both must be set, because the
    # floor cannot be computed without them. A 0th or 100th percentile is not a
    # floor — it keeps everything or one field-year — so the open interval.
    mode = dominance["floor_mode"]
    percentile, scope = dominance["floor_percentile"], dominance["floor_scope"]
    if mode == "absolute":
        if percentile is not None:
            problems.append(
                f"dominance.floor_mode is 'absolute' but dominance.floor_percentile is "
                f"{percentile}. Nothing reads the percentile under absolute mode, so leave "
                f"it null, or set floor_mode to 'percentile'."
            )
        if scope is not None:
            problems.append(
                f"dominance.floor_mode is 'absolute' but dominance.floor_scope is "
                f"'{scope}'. Nothing reads the scope under absolute mode, so leave it null."
            )
    else:
        if percentile is None:
            problems.append(
                "dominance.floor_mode is 'percentile' but dominance.floor_percentile is "
                "empty. Give the percentile (e.g. 25 for the 25th)."
            )
        elif not 0 < percentile < 100:
            problems.append(
                f"dominance.floor_percentile is {percentile}; it must lie strictly between "
                f"0 and 100, because the 0th percentile keeps every field-year and the "
                f"100th keeps one."
            )
        if scope is None:
            problems.append(
                "dominance.floor_mode is 'percentile' but dominance.floor_scope is empty. "
                "Say whether the percentile is taken over all field-years in the "
                "measurement years ('pooled') or within each window-year ('per_year')."
            )

    # The living inventors in Task 6 get pseudo-dates drawn from the same period
    # as the real deaths; a reference year outside it would not be comparable.
    task6 = values["task6"]
    if not (deaths["year_min"] <= task6["reference_year_min"] <= deaths["year_max"]) or \
       not (deaths["year_min"] <= task6["reference_year_max"] <= deaths["year_max"]):
        problems.append(
            f"task6.reference_year_min/max ({task6['reference_year_min']}-"
            f"{task6['reference_year_max']}) must lie inside the death window "
            f"({deaths['year_min']}-{deaths['year_max']}), or the living comparison "
            f"group is drawn from a different period than the treated."
        )

    if values["checkpoints"]["treated_workable_min"] > values["checkpoints"]["treated_comfortable"]:
        problems.append(
            "checkpoints.treated_workable_min is above checkpoints.treated_comfortable."
        )

    which = values["subsample"]["which"]
    if which == "pharma" and not values["subsample"]["pharma_cpc"]:
        problems.append("subsample.which is 'pharma' but subsample.pharma_cpc is empty.")
    if which == "semiconductors" and not values["subsample"]["semiconductor_cpc"]:
        problems.append(
            "subsample.which is 'semiconductors' but subsample.semiconductor_cpc is empty."
        )

    # Refuse a subsample that nothing implements. This key is validated just above and read by
    # NO step: setting it to 'pharma' would mint a new config hash, produce a fresh set of
    # stamped output filenames and a LATEST.md diff, and compute identical numbers on the full
    # sample. A change that appears to have worked in every respect except the results.
    #
    # That is rule 1 broken in the quiet direction, and it is the same defect item 9 found in
    # counting.field_size_method (C-38), fixed the same way: refuse the value rather than
    # ignore it. The keys stay, because the design commits to a pharma spotlight and a
    # semiconductor contrast and they are declared so nobody has to remember they are coming.
    # Delete this guard on the day a step actually reads them.
    if which != "all":
        problems.append(
            f"subsample.which is '{which}' and no step reads it, so the run would compute the "
            f"FULL sample while every output filename claimed a subsample that was never "
            f"applied. Set it back to 'all'. The pharma and semiconductor arms are declared "
            f"for later (docs/thesis_design.md 5.4) and are not built."
        )

    # The overlap brief's era starts must strictly increase, or two eras would share
    # years. Whether they fit THIS death window is checked by the step that reads
    # them (`eras()` in src/steps/05f_task_overlap.py), not here: a robustness config
    # that shortens the window for another step must still load. Likewise the lag
    # cutoffs' membership of the grids.                              [Yaren 2026-09-19]
    eras = values["task_overlap"]["era_start_years"]
    if not eras or any(later <= earlier for earlier, later in zip(eras, eras[1:])):
        problems.append(
            f"task_overlap.era_start_years is {eras}; it needs at least one start and the "
            f"starts must strictly increase, or two eras would share years."
        )

    return problems


# ---------------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------------

def _config_hash(values):
    """
    Eight characters that identify this exact set of analytical choices.

    Sorted keys, so the same settings always give the same hash no matter what
    order they were written in. The folder name and the file paths are left out
    because changing them cannot change a number.
    """
    flat = _flatten(values)
    analytical = {
        key: value for key, value in flat.items()
        if not any(key == skip or key.startswith(skip) for skip in NOT_IN_HASH)
    }
    canonical = json.dumps(analytical, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
