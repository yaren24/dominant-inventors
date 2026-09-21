"""
Where files go, how they are cached, and how outputs are stamped.

Why this file exists
--------------------
Three problems, all of which cost real time when they are not solved centrally.

1. Re-running a step should not redo work that is already done. A step is
   cached if its output exists, is newer than every input, and was produced by
   the same config. `--force` ignores all of that.

   The cache is deliberately conservative: ANY config change invalidates it,
   even one that could not possibly affect this particular file. A cache that is
   sometimes wrong is worse than one that sometimes recomputes.

2. Two configurations must never overwrite each other's results, and neither
   must two versions of the same configuration. So every output filename carries
   the 8-character config hash: changing a threshold produces a new file and
   leaves the old one intact.

3. Because that accumulates, each file is ALSO written under its plain name,
   refreshed every run, and `outputs/<config>/LATEST.md` records which hash is
   current and what changed since the previous run. When you need "the current
   version of Table 3" a year from now, that file tells you which one it is.
"""

import datetime
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from src.lib import config as config_module


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#
# The config states every data path relative to the repository root, as
# `data/raw/patentsview`, `data/interim` and so on. DATA_DIR replaces the
# leading `data/` for a reader whose download lives somewhere else — on an
# external disk, or on a cluster — so that no configuration file has to be
# edited to run this on another machine. Unset, the paths are what the config
# says, which is how every run reported in the thesis was produced.
#
# Outputs and logs are NOT redirected: they are results, not inputs, and they
# belong with the code that made them.

def data_root():
    """`data`, or DATA_DIR when the environment sets one."""
    return Path(os.environ.get("DATA_DIR") or "data")


def _resolve(configured):
    """Swap the leading `data/` of a configured path for DATA_DIR, when one is set."""
    parts = Path(configured).parts
    if parts and parts[0] == "data":
        return data_root().joinpath(*parts[1:])
    return Path(configured)


def check_data_is_there(cfg):
    """
    Stop with a readable message, rather than a traceback from inside DuckDB.

    A missing download is the first thing that goes wrong on a new machine, and
    the error it produces otherwise names a parquet file rather than the reason.
    """
    raw = _resolve(cfg.paths.raw_patentsview)
    if not raw.is_dir():
        where = os.environ.get("DATA_DIR")
        raise SystemExit(
            f"\nNo data found at {raw}/.\n\n"
            + (f"DATA_DIR is set to {where}.\n"
               if where else
               "DATA_DIR is not set, so paths are read from the config and are relative\n"
               "to the repository root. Point it at your own copy of the downloads with:\n"
               "    export DATA_DIR=/path/to/your/data\n")
            + "The downloads themselves are not redistributed with this code; the README\n"
              "names every source and where to get it.\n"
        )


def ensure_dirs(cfg):
    """Create the folders a run needs. They are absent from a fresh clone by design."""
    for folder in (interim_dir(cfg), processed_dir(cfg), Path(cfg.paths.logs), output_dir(cfg)):
        Path(folder).mkdir(parents=True, exist_ok=True)


def raw_patentsview_path(cfg, filename):
    return _resolve(cfg.paths.raw_patentsview) / filename


def raw_kjl_path(cfg, filename):
    return _resolve(cfg.paths.raw_kjl) / filename


def interim_dir(cfg):
    return _resolve(cfg.paths.interim)


def processed_dir(cfg):
    return _resolve(cfg.paths.processed)


def interim_path(cfg, filename):
    """Cached, rebuildable intermediate data. Safe to delete the whole folder."""
    return interim_dir(cfg) / filename


def processed_path(cfg, *parts):
    """Frozen artifacts, above all event_sample_v<N>/. Not safe to delete."""
    return processed_dir(cfg).joinpath(*parts)


def output_dir(cfg):
    return Path(cfg.paths.outputs) / cfg.name


def output_path(cfg, filename):
    """
    The permanent, hash-stamped name: task1_setup_note.md -> task1_setup_note_a3f19c02.md

    Never overwritten by a run with different settings, so an old result cannot
    silently disappear when a threshold changes.
    """
    stem = Path(filename)
    return output_dir(cfg) / f"{stem.stem}_{cfg.hash}{stem.suffix}"


def plain_output_path(cfg, filename):
    """The same file under its plain name, refreshed each run. This is the one you open."""
    return output_dir(cfg) / filename


# ---------------------------------------------------------------------------
# Superseding: what happens when the SAME config produces a DIFFERENT artefact
# ---------------------------------------------------------------------------
#
# Rule 7 keys outputs to the configuration, so a changed threshold produces a
# new hash and a new file and the old result survives. But an output depends on
# the config AND on the code that wrote it, and the hash cannot see the code.
# Correcting one sentence in a note therefore used to rewrite a hash-stamped
# file with different content under an unchanged stamp — the one thing rule 7
# exists to prevent — which is why three known-wrong sentences went uncorrected
# (notes/decisions.md C-31, C-32, C-33).
#
# So a stamped file is never overwritten with different content. It gains a
# version instead: task2_checkpoint1_dc0687cc.md, then ..._dc0687cc_v2.md. The
# plain copy always holds the newest, LATEST.md says what was superseded, and
# nothing that was ever sent to anyone disappears.

def version_of(path):
    """2 for ..._v2.md, 1 for the original. Used to order the versions."""
    stem = path.stem
    if "_v" not in stem:
        return 1
    tail = stem.rsplit("_v", 1)[1]
    return int(tail) if tail.isdigit() else 1


def stamped_versions(cfg, filename):
    """Every stamped copy of one output, oldest first."""
    stamped = output_path(cfg, filename)
    found = list(output_dir(cfg).glob(f"{stamped.stem}_v*{stamped.suffix}"))
    if stamped.exists():
        found.append(stamped)
    return sorted(found, key=version_of)


def current_stamped(cfg, filename):
    """
    The newest stamped copy, which is the one the plain file should match.

    Everything that compares the plain copy against "the pipeline's copy" has to
    ask this rather than output_path(), or the first supersession makes every
    later run report the file as hand-edited.
    """
    versions = stamped_versions(cfg, filename)
    return versions[-1] if versions else output_path(cfg, filename)


def _content_lines(text, cfg):
    """
    An output with the per-run noise removed, for deciding whether it changed.

    Two things differ between two runs of identical code and identical config,
    and neither is content: the `# generated:` line this file writes above a
    CSV, and any line naming the run's log folder, whose name carries the run
    timestamp. Everything else differing means the artefact really is different.
    """
    logs = str(cfg.paths.logs)
    return [line for line in text.splitlines()
            if not line.startswith("# generated:") and logs not in line]


def _stamped_target(cfg, filename, unchanged):
    """
    Where the permanent copy goes, or None when it is already there.

    None means "this run reproduced what is already on disk" — the common case,
    and the caller must then leave the file alone rather than rewrite identical
    bytes. Not fussiness: the modification time of a stamped artefact is how you
    check that a run did not disturb something already sent, which is exactly
    the check the last three sessions used.

    A supersession is not an error — usually it is a corrected sentence — but it
    must never be silent, because the file it replaces may already be in
    somebody's inbox.
    """
    stamped = output_path(cfg, filename)
    if unchanged:
        return None
    if not stamped.exists():
        return stamped

    previous = current_stamped(cfg, filename)
    target = stamped.with_name(
        f"{stamped.stem}_v{version_of(previous) + 1}{stamped.suffix}")
    print(f"  SUPERSEDED — {previous.name} is kept; this run wrote {target.name}.")
    print("    Same config hash, different content, so what changed is the code.")
    return target


def _write_both(cfg, filename, text, unchanged):
    """Write the plain copy, and the stamped copy if there is a new one. Returns both."""
    output_dir(cfg).mkdir(parents=True, exist_ok=True)
    plain = plain_output_path(cfg, filename)
    plain.write_text(text, encoding="utf-8")

    target = _stamped_target(cfg, filename, unchanged)
    if target is None:
        return current_stamped(cfg, filename), plain
    target.write_text(text, encoding="utf-8")
    return target, plain


def hand_edited_outputs(cfg):
    """
    Plain-name reports whose contents no longer match their stamped twin.

    Every report is written twice: once under a permanent, hash-stamped name and
    once under a plain one. They start identical, and only the plain one is ever
    rewritten, so if the two disagree the plain copy has been changed by
    something other than a step.

    This is not hypothetical. On 2026-07-30 an editor holding a stale copy of
    `task1_setup_note.md` saved it back over a freshly generated report — a
    version from an older config, with a stray `bk:` typed into the middle of a
    table. Nothing noticed. It was found by reading the file. The stamped twin
    was untouched and the fix was a one-line copy, which is the whole reason
    outputs are written twice.

    Returns a list of (plain_path, stamped_path). Empty is the normal case.
    """
    diverged = []
    for plain in sorted(output_dir(cfg).glob("*.md")):
        # LATEST.md is an index, not a report, and has no stamped twin.
        if plain.name == "LATEST.md" or cfg.hash in plain.stem:
            continue
        # The NEWEST version, not the original: after a supersession the plain
        # copy matches _v2, and comparing it against _v1 would report every
        # corrected report as hand-edited from then on.
        stamped = current_stamped(cfg, plain.name)
        if not stamped.exists():
            continue        # not produced under this config yet; nothing to compare
        if plain.read_text(encoding="utf-8") != stamped.read_text(encoding="utf-8"):
            diverged.append((plain, stamped))
    return diverged


def warn_about_hand_edited_outputs(cfg):
    """
    Say so, loudly, when a report has been edited outside the pipeline.

    A warning rather than an error: the stamped copy is authoritative and
    intact, so nothing downstream is wrong. But the plain copy is the one people
    open and email, so a silent hand edit is exactly the kind of thing that
    reaches an advisor.

    Called at the START of a step, before anything is rewritten, so the warning
    arrives while the edited text still exists rather than after it is replaced.
    """
    diverged = hand_edited_outputs(cfg)
    if not diverged:
        return diverged

    print("\n  WARNING — these reports differ from the copy the pipeline wrote:")
    for plain, stamped in diverged:
        print(f"    {plain.name}")
        print(f"      the pipeline's copy is {stamped.name}, which is intact")
    print("  Something edited them outside the pipeline — most often an editor")
    print("  saving an old copy back over a regenerated file. This run will")
    print("  overwrite them. If any of it was deliberate, save it elsewhere now.\n")
    return diverged


def write_output_text(cfg, filename, text):
    """
    Write a text output (a note, a table, a markdown file) both ways.

    Returns (stamped_path, plain_path). The stamped one is the record; the plain
    one is for reading and attaching to an email.

    If a stamped copy already exists holding different content — same config,
    changed code — this writes the next version rather than overwriting it. See
    the block above `version_of`.
    """
    existing = current_stamped(cfg, filename)
    unchanged = (existing.exists()
                 and _content_lines(existing.read_text(encoding="utf-8"), cfg)
                 == _content_lines(text, cfg))
    return _write_both(cfg, filename, text, unchanged)


def write_output_csv(cfg, filename, frame, step):
    """
    Write a results table both ways, with its provenance in comment lines above
    the header row.

    Rule 7 asks for the config name and hash in CSV headers rather than only in
    filenames, because a CSV gets emailed, renamed and opened in Excel, and any
    of those separates it from the name that said which analysis produced it.
    The comment lines survive all three.

    They are read transparently by `pandas.read_csv(path, comment="#")`, by R's
    `read.csv(comment.char="#")` and by Stata's `import delimited`.

    Deliberately NOT covered by the hand-edit detector above, which watches only
    markdown. A CSV is machine output that nobody edits by hand, and a
    spreadsheet round-trip reformats every number, so including them would flag
    the file on every run and train the reader to ignore the warning.
    """
    header = "\n".join([
        f"# config: {cfg.name}",
        f"# config_hash: {cfg.hash}",
        f"# written_by: {step}",
        f"# generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
    ])
    text = header + "\n" + frame.to_csv(index=False)

    # The `# generated:` line differs on every run and is excluded from the
    # comparison, so a rerun that changes nothing does not manufacture a _v2.
    existing = current_stamped(cfg, filename)
    unchanged = (existing.exists()
                 and _content_lines(existing.read_text(encoding="utf-8"), cfg)
                 == _content_lines(text, cfg))
    return _write_both(cfg, filename, text, unchanged)


def write_output_figure(cfg, fig, filename, **savefig_options):
    """
    Write a figure both ways, under the same no-overwrite rule as a note.

    This exists because `save_figure` in 02b promised in its docstring that the
    hashed copy is never overwritten and then overwrote it: matplotlib is handed
    a path and writes to it. The six Checkpoint 1 figures are the artefacts that
    were actually sent to the advisor, so that was the worst place to have the
    guarantee be untrue.

    The plain copy is written first and then compared byte for byte, which needs
    no rendering twice and no temporary file. Figures here carry no embedded
    timestamp, so identical code and identical data give identical bytes — a
    property worth knowing, since it is what makes the comparison meaningful.
    """
    output_dir(cfg).mkdir(parents=True, exist_ok=True)
    plain = plain_output_path(cfg, filename)
    fig.savefig(plain, **savefig_options)
    drawn = plain.read_bytes()

    existing = current_stamped(cfg, filename)
    unchanged = existing.exists() and existing.read_bytes() == drawn
    target = _stamped_target(cfg, filename, unchanged)
    if target is None:
        return existing, plain
    target.write_bytes(drawn)
    return target, plain


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

def read_parquet(con, path):
    """Open a Parquet file as a DuckDB relation. Nothing is loaded into memory yet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. If this is a converted bulk table, run:\n"
            f"    python src/01_convert_to_parquet.py"
        )
    return con.sql(f"SELECT * FROM read_parquet('{path.as_posix()}')")


def write_parquet(con, frame, path, cfg, inputs=(), step=""):
    """
    Write a DuckDB relation or a pandas frame to Parquet, with a note beside it.

    The note (<file>.meta.json) records which config produced the file and from
    what, which is what makes the cache check below trustworthy.

    The COPY goes to a scratch name in the same directory and is moved onto the
    real one only once it has finished. DuckDB does this itself when it is
    REPLACING a file, and not when it is writing a name that does not exist yet:
    then it grows the destination in place. So a run that is killed rather than
    raising leaves a truncated file wearing the artifact's exact name — there is
    one from 24 August 2026, 660 MB of 01_citations.parquet with no footer. An
    out-of-memory kill leaves no traceback and the harness reports it as a clean
    exit, so nothing else in the pipeline would have said so. data/interim/ has
    no versioned copy to fall back on the way outputs/ does under rule 7, and the
    .meta.json is not this guard: it makes the CACHE refuse the file, one step
    later and only if a step gets that far.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Beside the file, not in a temp directory: os.replace is only atomic within
    # one filesystem, and data/interim/ is not always on the same disk as /tmp.
    scratch = path.with_name(f"unfinished_{path.name}")
    destination = scratch.as_posix().replace("'", "''")

    if isinstance(frame, duckdb.DuckDBPyRelation):
        frame.to_view("frame_to_write", replace=True)
    elif isinstance(frame, pd.DataFrame):
        con.register("frame_to_write", frame)
    else:
        raise TypeError(f"Cannot write {type(frame).__name__} to Parquet.")

    try:
        con.execute(
            f"COPY (SELECT * FROM frame_to_write) TO '{destination}' "
            f"(FORMAT parquet, COMPRESSION zstd)"
        )
        n_rows = con.sql(f"SELECT count(*) FROM read_parquet('{destination}')").fetchone()[0]
    except BaseException:
        # Covers the gap between a COPY that finished and the rename below —
        # the row count re-reads the file and could fail on it. When the COPY
        # ITSELF raises, DuckDB has already removed what it wrote, so this is a
        # no-op there; and a killed process runs no handler at all, which is why
        # the scratch name rather than this line is what protects the artifact.
        scratch.unlink(missing_ok=True)
        raise

    # The note goes away BEFORE the data is moved into place, and is written
    # again after. A run killed in that window leaves the new file with no note
    # at all, and built_by_this_config reads the note to decide whether a file
    # may be read — no note means it returns False and
    # require_built_by_this_config refuses to run, naming the rebuild. Left in
    # the other order, the window leaves complete new data described by the
    # PREVIOUS run's note, which is silent where the truncated file this
    # replaced was loud. Absent is a safe state. Stale is not.
    _meta_path(path).unlink(missing_ok=True)
    scratch.replace(path)

    _meta_path(path).write_text(
        json.dumps(
            {
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "step": step,
                "config_name": cfg.name,
                "config_hash": cfg.hash,
                "rows": n_rows,
                "inputs": [Path(i).as_posix() for i in inputs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

def is_cached(path, inputs, cfg, force=False):
    """
    True when `path` can be reused instead of rebuilt.

    Three conditions, all required: the file exists; it was produced by this
    exact config; and no input has changed since it was written.
    """
    path = Path(path)
    if force or not path.exists():
        return False

    meta_path = _meta_path(path)
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("config_hash") != cfg.hash:
        return False

    output_time = path.stat().st_mtime
    for source in inputs:
        source = Path(source)
        if not source.exists() or source.stat().st_mtime > output_time:
            return False
    return True


def built_by_this_config(path, cfg):
    """
    True when `path` was written by a run of THIS exact configuration.

    Different from is_cached: that question is "may I skip work?", this one is
    "am I allowed to read this at all?". A step that reads an earlier step's
    output has to ask the second question, because the filenames in
    data/interim/ do not carry the config hash — 01_patents.parquet is
    01_patents.parquet whether it holds one CPC section or all of them. Reading
    a sample-mode file during a full run would produce a complete set of
    plausible results for a thirteenth of the data, and nothing would look wrong.
    """
    path = Path(path)
    meta_path = _meta_path(path)
    if not path.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("config_hash") == cfg.hash


def require_built_by_this_config(paths, cfg, rebuild_with):
    """Refuse to run on another configuration's files. Raises, naming the fix."""
    stale = [Path(p) for p in paths if not built_by_this_config(p, cfg)]
    if not stale:
        return
    listed = "\n".join(f"    {p}" for p in stale)
    raise SystemExit(
        f"\nThese inputs were not built by the current config (hash {cfg.hash}):\n{listed}\n\n"
        f"They are missing, or they were produced by different settings — most often by a "
        f"sample-mode run when the config now says sample_mode: {cfg.sample_mode}. Reading "
        f"them would give a full set of results computed on the wrong data, and nothing would "
        f"look wrong.\n\nRebuild them first:\n    {rebuild_with}\n"
    )


def read_meta(path):
    """
    The note beside a Parquet file, or an empty dict when there is none.

    Public because the note is the only record of WHICH configuration built a file
    in data/interim/, whose filenames carry no hash. A step comparing two arms has
    to be able to say which hash the other arm was built at, and reading it off the
    file beats configuring it: a hash in a config is a claim, this is a fact.
    """
    meta_path = _meta_path(Path(path))
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def cache_message(path):
    """The line a step prints when it skips work, so skipping is never silent."""
    meta = read_meta(path)
    if meta:
        return (f"  cached   {Path(path).name}  ({meta.get('rows', '?'):,} rows, built "
                f"{meta.get('created', '?')}, config {meta.get('config_hash', '?')}) "
                f"— use --force to rebuild")
    return f"  cached   {Path(path).name} — use --force to rebuild"


def _meta_path(path):
    return Path(str(path) + ".meta.json")


# ---------------------------------------------------------------------------
# LATEST.md — which file is the current one
# ---------------------------------------------------------------------------


def write_latest(cfg, step, files, summary=""):
    """
    Rewrite outputs/<config_name>/LATEST.md.

    Says which hash is current, when it ran, which config values differ from the
    previous run, and which files that run produced. The comparison is against
    `.last_run.json`, which this function then updates.

    It also keeps a PER-STEP record, because the paragraph above is not enough on its
    own. Rule 7 promises that when you need "the current version of Table 3", this file
    tells you which one it is — and until 2026-08-09 it described only the LAST STEP RUN.
    So after running 02b it listed 02b's seven files and said nothing about Task 3's, and
    a reader could not tell that a report predated the current hash.

    That is not hypothetical: C-42 D81 records the Checkpoint 1 note sitting three days
    stale against its own config, with nothing saying so, because 02b had not been re-run
    since a later step moved the hash. The per-step table below is the fix, and a step
    whose hash no longer matches the current one is marked STALE by name.
    """
    folder = output_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / ".last_run.json"

    current = config_module.flatten(cfg)
    differences = []
    steps = {}
    if state_path.exists():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        steps = previous_state.get("steps", {})
        previous = previous_state.get("config_values", {})
        if previous_state.get("config_hash") != cfg.hash:
            for key in sorted(set(previous) | set(current)):
                was, now = previous.get(key, "(absent)"), current.get(key, "(absent)")
                if was != now:
                    differences.append((key, was, now))

    steps[step] = {
        "config_hash": cfg.hash,
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "files": [Path(produced).name for produced in files],
    }

    lines = [
        f"# LATEST — outputs for config `{cfg.name}`",
        "",
        f"**Current config hash: `{cfg.hash}`**",
        "",
        f"- Last step run: `{step}`",
        f"- When: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- Config file: `{cfg.source_path}`",
        "",
    ]
    if summary:
        lines += [summary, ""]

    lines += ["## Files from this run", ""]
    superseded = []
    for produced in files:
        name = Path(produced).name
        version = version_of(Path(produced))
        if version > 1:
            lines.append(f"- `{name}` — **supersedes** an earlier copy under the same hash")
            superseded.append(name)
        else:
            lines.append(f"- `{name}`")
    lines += [
        "",
        "Each of these also exists under a plain name without the hash — same content, "
        "overwritten on every run. The hashed copies are never overwritten.",
        "",
    ]
    if superseded:
        lines += [
            "A `_v2` (or later) means this run produced different content under an "
            "UNCHANGED config hash, so what changed was the code that writes the file "
            "rather than a setting. The earlier copy is kept beside it and is still the "
            "one that was sent, if it was sent. See notes/decisions.md C-34.",
            "",
        ]

    # The per-step table. This is the section that answers "which is the current version of
    # Table 3" for a step you did NOT just run, and it names a stale step rather than leaving
    # a reader to compare timestamps.
    stale = [name for name, record in steps.items() if record["config_hash"] != cfg.hash]
    lines += ["## Every step, and the hash it last wrote at", ""]
    lines += ["| step | last written | at hash | files | |",
              "| --- | --- | --- | --: | --- |"]
    for name in sorted(steps):
        record = steps[name]
        marker = "**STALE**" if name in stale else "current"
        lines += [f"| `{name}` | {record['when'][:16].replace('T', ' ')} | "
                  f"`{record['config_hash']}` | {len(record['files'])} | {marker} |"]
    lines += [""]
    if stale:
        lines += [
            f"**{len(stale)} step(s) have not been re-run since the config hash moved to "
            f"`{cfg.hash}`.** Their outputs on disk were written under an earlier hash, so any "
            f"report quoting them is quoting a number from a different configuration. Re-run "
            f"them, or treat their files as historical: "
            + ", ".join(f"`{name}`" for name in sorted(stale)) + ".",
            "",
            "This section exists because that happened silently once — see "
            "notes/decisions.md C-42 D81, where the Checkpoint 1 note sat three days stale "
            "against its own config and nothing said so.",
            "",
        ]
    else:
        lines += ["Every step that has ever run for this config ran at the current hash.", ""]

    lines += ["## What changed since the previous run", ""]
    if not state_path.exists():
        lines.append("First recorded run for this config.")
    elif not differences:
        lines.append("Nothing. Same config hash as the previous run.")
    else:
        lines += ["| setting | was | now |", "| --- | --- | --- |"]
        lines += [f"| `{key}` | `{was}` | `{now}` |" for key, was, now in differences]

    (folder / "LATEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    state_path.write_text(
        json.dumps(
            {
                "config_hash": cfg.hash,
                "step": step,
                "when": datetime.datetime.now().isoformat(timespec="seconds"),
                "config_values": current,
                "steps": steps,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return folder / "LATEST.md"
