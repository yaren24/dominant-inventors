# Dominant inventors and the fields they occupy

## What this is

My MA thesis asks whether a technology field opens up or contracts after the premature
death of the inventor who dominated it. This repository builds the dataset it needs: USPTO
patent, citation and CPC records — 152.6 M citation rows and 59.8 M classification rows —
linked to the Kaltenberg–Jaffe–Lachman inventor-death file and turned into inventor × field
× year panels of share, rank and margin. It ships with 70 tests that run without any of the
downloaded data.

## Data

**No raw data is redistributed here.** Both sources are public; download them into
`$DATA_DIR/raw/`. PatentsView re-disambiguates inventors with every release, so the vintage
below is part of the specification, not a detail.

| Source | What it holds | Size |
|---|---|---|
| PatentsView bulk tables (USPTO, 2026-04-10 release) | granted patents, disambiguated inventors and assignees, CPC codes, citations, filing dates | 9.45 M patents, 24.0 M inventor–patent rows, 59.8 M CPC rows, 152.6 M citations |
| PatentsView persistent inventor crosswalk | the inventor ID each slot held in every past release, one column per release | 24.1 M rows, 30 columns |
| Kaltenberg, Jaffe & Lachman, *The Age of Invention* (NBER WP 28768; Harvard Dataverse, CC0) | inventor birth years and death dates, matched to patents 1976–2018 | 1.86 M inventors with a birth year; 535,120 with a death record of any year, before linking |

## Pipeline

| script | input → what it does → output |
|---|---|
| `src/01_convert_to_parquet.py` | the downloaded `.zip` TSVs → a format change only; every column is read as text, so no bad date is silently coerced → Parquet |
| `src/02_build_patent_tables.py` | those tables and the KJL CSVs → applies the patent-universe rules, assigns patents to CPC fields, and links KJL's inventor IDs to current PatentsView ones → eight spine tables, including the link, birth years, and 402,205 usable deaths (linked to a measurable inventor, one death per person, 1986–2016) |
| `src/03_measure_dominance.py` | the spine → per inventor × field × year, the share of the field's patents filed in a five-year window, the rank and two margins over the runner-up, at three CPC widths → inventor-level and field-level panels carrying the small-field floor |
| `src/04_select_dominant_inventors.py` | those panels → applies the dominance cutoffs. This is the only script that applies one: shares, ranks and margins are stored raw upstream, so a cutoff can be changed without rebuilding anything → the candidate pool, with persistence and field-size breakdowns |

Extracted from a larger thesis repository. Comments throughout cite design notes, steps and
configs that live there; the legend at the top of `config/default.yaml` says so.

## Design decisions

**How a field is defined.** CPC subclass is the baseline, with class and main group
reported beside it — field width changes everyone's share mechanically, so all three are
carried through rather than one being chosen. Patents are classified by the *current* CPC
table, not by the classification they carried at grant. Freezing the old classification at
the death and then watching the field afterwards would count the wrong patents into that
field: reclassification is how the office records what a patent turned out to be about. The
cost is a look-ahead — a patent can enter the field after the death — and it is accepted
because the alternative is unusable. Classification at issue covers none of the filings
before 2000, 4.0% of the 2000s and 86.3% of the 2010s, against 100% for the current table,
so using it would cost most of the death window.

**How long a window.** Shares are measured over the field's patents filed in years t−4
through t. Five years is the window used in this literature, and it is the right shape:
inventive work takes time to appear, so the measure looks backwards from the year in
question rather than around it.

**How a patent is credited.** A patent with n inventors counts 1/n for each. Shares within
a field then sum to 1, and someone on large teams cannot accumulate dominance by being one
of many. Full counting is a reported robustness check, not a footnote: it names a different
top inventor in 53% to 57% of field-years depending on field width.

**The small-field floor.** Owning 30% of a field with ten patents is not dominance of
anything, so field-years below a floor are excluded. The 100-patent floor in the default
config is arbitrary — a working number for looking at the data, not a finding. Every table
is therefore produced both with and without it, and percentile floors are reported
alongside; the 25th percentile is the likely final choice.

**Linking deaths to inventors.** Kaltenberg–Jaffe–Lachman name a person by a single patent
appearance, and PatentsView re-runs its inventor disambiguation with every release, so the
two files do not share an identifier. The link is built two ways: from USPTO's own
persistent crosswalk, which records both identities on one row, and by reconstructing it
from the inventor's position on the patent. The crosswalk is USPTO's own matching and is
the safer of the two, so it is preferred; the second route is kept in order to measure how
far they disagree, which is 27,255 of 1,858,356 identities, or 1.5%.

**What counts as a premature death.** Age under 60 at death, the threshold this literature
uses. Age 65 is carried as a robustness check. Cause of death is not used — establishing it
would need an external source, and that is a later addition rather than part of this build.

**The prosecution window.** In the citation panel, which a later step builds: a patent's
reference list is not closed when the patent is filed; it closes when it is granted, and
between 43% and 71% of a field's window patents are still in prosecution at the measurement
date. A citation-based dose measured before a death is therefore built partly from
references created after it, in a literature that exists because the death changed how
people cite — reverse causality in the regressor, and it does not attenuate. The project
keeps filing as the dating convention everywhere and reports the result again on the
sub-sample whose reference lists provably closed before the death, rather than claiming the
leak away. A second and smaller rule follows from the same fact: a patent cannot be cited
before it is granted, so edges that could not physically have existed at the measurement
date are excluded — between 1.1% and 4.4% of them, depending on the year.

## Running it

Python 3.11 or newer; last run on 3.14.3.

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
export DATA_DIR=/path/to/your/data        # if unset, paths default to ./data

.venv/bin/python src/01_convert_to_parquet.py
.venv/bin/python src/02_build_patent_tables.py
.venv/bin/python src/03_measure_dominance.py
.venv/bin/python src/04_select_dominant_inventors.py

.venv/bin/python -m pytest -q             # 70 tests; these need no downloaded data
```

On an Apple M1 with 8 GB of RAM, `02` takes about six minutes, `03` about seven, and `04`
half a minute. `01` is a one-off conversion and depends mostly on disk.

Every analytical choice lives in `config/default.yaml`; the code reads it and hardcodes
nothing. The file is the thesis's full configuration; the scripts here read its
build-and-dominance sections. `--config` runs a different one, `--force` ignores the cache,
`sample_mode: true` restricts the run to one CPC section.

## Status

Estimation is in progress. This repository covers the data construction.
