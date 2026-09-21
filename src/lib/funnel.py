"""
The sample-size tracker: every row this analysis drops, and why.

Why this file exists
--------------------
A funnel table is not documentation written after the fact. It is the methods
section of the thesis, produced by the code that actually ran, so it cannot
disagree with what was done. Reviewers ask "how did 8 million patents become
40,000 inventor-field pairs?" and the honest answer has to be reconstructible
line by line.

So no filtering happens inline anywhere in this repository. Every row that
disappears goes through `funnel.filter`, which records how many rows, how many
distinct patents and how many distinct inventors were lost, together with a
mandatory sentence explaining why. Those sentences are yours to write, and they
end up in the thesis nearly unchanged — write them as prose, not as notes.

Conditions are SQL
------------------
`condition` is always SQL, for pandas frames and DuckDB relations alike, so
there is one syntax to remember rather than two. It states what is KEPT, not
what is dropped:

    df = funnel.filter(df, "utility_only", "patent_type = 'utility'", why="...")

What a run leaves behind
------------------------
    logs/runs/<timestamp>_<step>/
        funnel.csv         one row per step, machine readable
        data_diary.md      the same thing as prose, with your why-sentences
        config_used.yaml   the config for this run, byte for byte
        session.log        timings and counts, written as the run proceeds, so
                           a crash still leaves a trail

Usage
-----
    funnel = Funnel("02_build_patent_tables", cfg)
    df = funnel.filter(df, "utility_only", "patent_type = 'utility'", why="...")
    funnel.checkpoint("after the patent universe is fixed", df)
    funnel.note("Application dates before 1900 are kept for now; see step 03.")
    funnel.finish()
"""

import csv
import datetime
import time
from pathlib import Path

import duckdb
import pandas as pd


# A why-sentence shorter than this is a note to self, not an explanation. The
# number is here rather than in the config because it is about writing
# discipline, not about the analysis.
MINIMUM_WHY_CHARACTERS = 20


class Funnel:
    """Records every row-dropping step of one run of one step script."""

    def __init__(self, step_name, cfg, started_at=None):
        self.step_name = step_name
        self.cfg = cfg
        self.started_at = started_at or datetime.datetime.now()
        self.steps = []

        stamp = self.started_at.strftime("%Y-%m-%d_%H%M%S")
        self.run_dir = Path(cfg.paths.logs) / f"{stamp}_{step_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # One connection for the whole run. It is also what lets a pandas frame
        # be filtered with the same SQL as a DuckDB relation.
        self._con = duckdb.connect()

        # Counting rows means scanning, and DuckDB relations are lazy, so the
        # same scan would be repeated for every step in a chain. When the frame
        # handed in is the very object the previous step handed back, its counts
        # are already known.
        self._last_output = None
        self._last_counts = None

        self._log(f"run started: step={step_name} config={cfg.name} hash={cfg.hash}")
        self._log(f"config file: {cfg.source_path}")

    # -----------------------------------------------------------------------
    # The three things a step script calls
    # -----------------------------------------------------------------------

    def filter(self, frame, name, condition, why):
        """
        Keep the rows matching `condition` (SQL), and record what that cost.

        name       a short label, e.g. "utility_only"
        condition  SQL for what is KEPT, e.g. "patent_type = 'utility'"
        why        a full sentence for the methods section. Mandatory.
        """
        if not why or len(why.strip()) < MINIMUM_WHY_CHARACTERS:
            raise ValueError(
                f"Step '{name}' has no real explanation. Every filter needs a why-sentence, "
                f"because these sentences become the methods section of the thesis. Write "
                f"what was dropped and what the economic or data reason was."
            )

        clock = time.perf_counter()
        before = self._counts(frame)
        result = self._apply(frame, condition)
        after = self._counts(result)
        elapsed = time.perf_counter() - clock

        self._record("filter", name, condition, why, before, after, elapsed)
        self._last_output, self._last_counts = result, after
        return result

    def checkpoint(self, name, frame):
        """Record where the sample stands, without dropping anything."""
        clock = time.perf_counter()
        counts = self._counts(frame)
        self._record("checkpoint", name, "", "", counts, counts,
                     time.perf_counter() - clock)
        return frame

    def note(self, text):
        """A free-text remark for the data diary — something noticed, not something dropped."""
        self._record("note", "", "", text, None, None, 0.0)
        self._log(f"note: {text}")

    # -----------------------------------------------------------------------
    # Finishing
    # -----------------------------------------------------------------------

    def finish(self, print_to_terminal=True):
        """Write funnel.csv, data_diary.md, config_used.yaml and session.log. Returns the folder."""
        self._write_csv()
        self._write_diary()
        # Byte for byte, so that the run can be reproduced from this file alone.
        (self.run_dir / "config_used.yaml").write_text(self.cfg.source_text, encoding="utf-8")

        total = (datetime.datetime.now() - self.started_at).total_seconds()
        self._log(f"run finished in {total:.1f}s; {len(self.steps)} entries")

        if print_to_terminal:
            print(self.as_text())
            print(f"Full log: {self.run_dir}")
        return self.run_dir

    def as_text(self):
        """The funnel as a plain table, for the terminal."""
        header = (f"\nFunnel — {self.step_name}  (config '{self.cfg.name}', hash {self.cfg.hash})\n"
                  f"{'step':<28} {'rows before':>14} {'rows after':>14} {'dropped':>14}\n"
                  f"{'-' * 72}")
        lines = [header]
        for entry in self.steps:
            if entry["kind"] == "note":
                lines.append(f"  note: {entry['why'][:66]}")
                continue
            dropped = entry["rows_before"] - entry["rows_after"]
            share = f"{dropped / entry['rows_before']:.1%}" if entry["rows_before"] else "—"
            label = entry["name"] if entry["kind"] == "filter" else f"[{entry['name']}]"
            lines.append(
                f"{label:<28} {entry['rows_before']:>14,} {entry['rows_after']:>14,} "
                f"{dropped:>10,} {share:>6}"
            )
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _apply(self, frame, condition):
        """Run one SQL condition against a pandas frame or a DuckDB relation."""
        if isinstance(frame, duckdb.DuckDBPyRelation):
            return frame.filter(condition)
        if isinstance(frame, pd.DataFrame):
            # DuckDB does the filtering here too, so that the same SQL works for
            # both kinds of input and nobody has to remember two dialects.
            self._con.register("funnel_input", frame)
            result = self._con.sql("SELECT * FROM funnel_input WHERE " + condition).df()
            self._con.unregister("funnel_input")
            return result
        raise TypeError(
            f"funnel.filter needs a pandas DataFrame or a DuckDB relation, "
            f"not {type(frame).__name__}."
        )

    def _counts(self, frame):
        """Rows, distinct patents and distinct inventors — in one pass over the data."""
        if frame is self._last_output and self._last_counts is not None:
            return self._last_counts

        patent_column = self.cfg.columns.patent_id
        inventor_column = self.cfg.columns.inventor_id
        available = self._column_names(frame)

        pieces = ["count(*) AS rows"]
        pieces.append(f'count(DISTINCT "{patent_column}") AS patents'
                      if patent_column in available else "NULL AS patents")
        pieces.append(f'count(DISTINCT "{inventor_column}") AS inventors'
                      if inventor_column in available else "NULL AS inventors")
        aggregate = ", ".join(pieces)

        if isinstance(frame, duckdb.DuckDBPyRelation):
            rows, patents, inventors = frame.aggregate(aggregate).fetchone()
        else:
            self._con.register("funnel_input", frame)
            rows, patents, inventors = self._con.sql(
                f"SELECT {aggregate} FROM funnel_input").fetchone()
            self._con.unregister("funnel_input")

        return {"rows": rows, "patents": patents, "inventors": inventors}

    @staticmethod
    def _column_names(frame):
        if isinstance(frame, duckdb.DuckDBPyRelation):
            return list(frame.columns)
        if isinstance(frame, pd.DataFrame):
            return list(frame.columns)
        raise TypeError(f"Expected a DataFrame or a DuckDB relation, got {type(frame).__name__}.")

    def _record(self, kind, name, condition, why, before, after, elapsed):
        entry = {
            "order": len(self.steps) + 1,
            "kind": kind,
            "name": name,
            "condition": condition,
            "why": why,
            "rows_before": before["rows"] if before else None,
            "rows_after": after["rows"] if after else None,
            "patents_before": before["patents"] if before else None,
            "patents_after": after["patents"] if after else None,
            "inventors_before": before["inventors"] if before else None,
            "inventors_after": after["inventors"] if after else None,
            "seconds": round(elapsed, 3),
        }
        self.steps.append(entry)
        if kind != "note":
            self._log(
                f"{kind} {name}: rows {entry['rows_before']:,} -> {entry['rows_after']:,} "
                f"({elapsed:.1f}s)"
            )

    def _log(self, message):
        """Append one line to session.log immediately, so a crash still leaves a trail."""
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(self.run_dir / "session.log", "a", encoding="utf-8") as handle:
            handle.write(f"{stamp}  {message}\n")

    def _write_csv(self):
        columns = ["order", "kind", "name", "condition", "rows_before", "rows_after",
                   "rows_dropped", "share_dropped", "patents_before", "patents_after",
                   "inventors_before", "inventors_after", "seconds", "why"]
        with open(self.run_dir / "funnel.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for entry in self.steps:
                row = dict(entry)
                if entry["rows_before"] is not None:
                    dropped = entry["rows_before"] - entry["rows_after"]
                    row["rows_dropped"] = dropped
                    row["share_dropped"] = (round(dropped / entry["rows_before"], 6)
                                            if entry["rows_before"] else "")
                else:
                    row["rows_dropped"] = row["share_dropped"] = ""
                writer.writerow({key: row.get(key, "") for key in columns})

    def _write_diary(self):
        """The funnel as prose. This is the draft of the methods section."""
        lines = [
            f"# Data diary — {self.step_name}",
            "",
            f"- Run: `{self.run_dir.name}`",
            f"- Config: `{self.cfg.name}`, hash `{self.cfg.hash}`, from `{self.cfg.source_path}`",
            f"- Started: {self.started_at:%Y-%m-%d %H:%M:%S}",
            "",
            "Written by `src/lib/funnel.py` from the run itself, so it cannot disagree with "
            "what the code did. Every paragraph below is one step of the sample construction.",
            "",
            "---",
            "",
        ]
        for entry in self.steps:
            if entry["kind"] == "note":
                lines += [f"> **Note.** {entry['why']}", ""]
                continue

            if entry["kind"] == "checkpoint":
                lines += [
                    f"### Checkpoint: {entry['name']}",
                    "",
                    f"{entry['rows_after']:,} rows"
                    + _distinct_phrase(entry["patents_after"], entry["inventors_after"])
                    + ".",
                    "",
                ]
                continue

            dropped = entry["rows_before"] - entry["rows_after"]
            share = f"{dropped / entry['rows_before']:.1%}" if entry["rows_before"] else "n/a"
            lines += [
                f"### {entry['order']}. {entry['name']}",
                "",
                entry["why"],
                "",
                f"Kept rows where `{entry['condition']}`. "
                f"Rows {entry['rows_before']:,} → {entry['rows_after']:,} "
                f"({dropped:,} dropped, {share}).",
                "",
            ]
            detail = _change_phrase("Distinct patents", entry["patents_before"],
                                    entry["patents_after"])
            detail += _change_phrase("distinct inventors", entry["inventors_before"],
                                     entry["inventors_after"])
            if detail:
                lines += [detail.strip().capitalize(), ""]

        (self.run_dir / "data_diary.md").write_text("\n".join(lines), encoding="utf-8")


def _distinct_phrase(patents, inventors):
    parts = []
    if patents is not None:
        parts.append(f"{patents:,} distinct patents")
    if inventors is not None:
        parts.append(f"{inventors:,} distinct inventors")
    return (", " + ", ".join(parts)) if parts else ""


def _change_phrase(label, before, after):
    if before is None or after is None:
        return ""
    return f"{label} {before:,} → {after:,}. "
