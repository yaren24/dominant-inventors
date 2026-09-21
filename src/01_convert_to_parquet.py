"""
Step 01 — convert the downloaded PatentsView bulk tables from TSV to Parquet.

Reads : data/raw/patentsview/*.zip   (exactly as downloaded, never modified)
Writes: data/interim/*.parquet

Why this step exists
--------------------
The bulk tables ship as tab-separated text. Parquet is columnar and compressed, so it is
several times smaller on disk and much faster to query, because DuckDB reads only the
columns a query actually asks for instead of parsing every line of text.

Why everything is loaded as text
--------------------------------
PatentsView contains MySQL zero-dates ("0000-00-00") and other values that are not valid
dates. If we let the reader guess column types, it crashes on those rows or silently turns
them into nulls. Loading every column as text means nothing is lost or altered here.
Converting dates and numbers happens later, in step 02, where a failed conversion is
visible and recorded in the funnel rather than hidden inside a file-format step.

This step therefore drops nothing and changes nothing. It is a format change only.

Usage
-----
    python src/01_convert_to_parquet.py
    python src/01_convert_to_parquet.py --force     # redo files already converted
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import duckdb

def sql_quote(value: str) -> str:
    """Wrap a path in single quotes for inlining into SQL, doubling any quote inside it."""
    return "'" + value.replace("'", "''") + "'"


# This step reads no config — it is a format change, and it runs before the
# config has anything to say. DATA_DIR is therefore its only path setting;
# unset, it is `data/` relative to the repository root, which is where the
# config expects the downloads to be.
DATA_DIR = Path(os.environ.get("DATA_DIR") or "data")
RAW_DIR = DATA_DIR / "raw" / "patentsview"
OUT_DIR = DATA_DIR / "interim"


def convert_one(zip_path: Path, out_dir: Path, force: bool = False) -> None:
    """Unzip one bulk table into a temporary folder and write it out as Parquet."""
    out_path = out_dir / (zip_path.stem + ".parquet")

    if out_path.exists() and not force:
        print(f"  skip     {zip_path.name}  (already converted; use --force to redo)")
        return

    # Unzip to a temporary directory so the raw folder is never written to.
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".tsv", ".csv"))]
            if len(names) != 1:
                raise RuntimeError(
                    f"{zip_path.name} contains {len(names)} data files, expected 1: {names}"
                )
            tsv_path = Path(z.extract(names[0], tmp))

        con = duckdb.connect()

        # Read the file in whatever order is fastest, and spill to disk rather than RAM.
        #
        # By default DuckDB keeps the output rows in the same order as the input, which
        # means holding the whole table in memory before writing it. That is affordable
        # for a 1 GB table and not for g_persistent_inventor, which is 10.3 GB of text.
        # The order rows sit in inside a Parquet file carries no meaning here — every
        # query downstream joins and groups, and none of them reads the file top to
        # bottom — so nothing is lost by giving it up. Contents are unaffected.
        con.execute("SET preserve_insertion_order = false")
        temp_dir = out_dir / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = {sql_quote(str(temp_dir))}")

        # DuckDB does not accept bound parameters for a COPY destination, so the paths are
        # written into the statement. sql_quote doubles any single quote in a path.
        src = sql_quote(str(tsv_path))
        dst = sql_quote(str(out_path))
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_csv(
                    {src},
                    delim        = '\t',
                    header       = true,
                    quote        = '"',
                    escape       = '"',
                    all_varchar  = true,   -- keep everything as text; see docstring
                    sample_size  = -1,     -- scan the whole file, do not guess from the top
                    null_padding = true    -- tolerate rows with missing trailing fields
                )
            ) TO {dst} (FORMAT parquet, COMPRESSION zstd)
            """
        )

        # Report the sizes, so the space saving is visible and the row count is on record.
        n_rows = con.execute(
            f"SELECT count(*) FROM read_parquet({dst})"
        ).fetchone()[0]
        tsv_mb = tsv_path.stat().st_size / 1e6
        pq_mb = out_path.stat().st_size / 1e6
        con.close()

    shrinkage = f"  ({tsv_mb / pq_mb:.1f}x smaller)" if pq_mb > 0 else ""
    print(
        f"  ok       {zip_path.name:<40} {n_rows:>12,} rows   "
        f"{tsv_mb:>8.1f} MB tsv -> {pq_mb:>7.1f} MB parquet{shrinkage}"
    )


def where_data_dir() -> str:
    """One line saying where this run is looking, for the message when it finds nothing."""
    set_to = os.environ.get("DATA_DIR")
    if set_to:
        return f"DATA_DIR is set to {set_to}."
    return "DATA_DIR is not set, so paths are relative to the repository root."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="reconvert existing files")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(raw_dir.glob("*.zip"))
    if not zips:
        sys.exit(
            f"No .zip files found in {raw_dir}.\n"
            f"{where_data_dir()}\n"
            "The bulk tables are not redistributed with this code; the README names\n"
            "every source and where to download it."
        )

    # Fail early if the disk is obviously too small, rather than half way through.
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    print(f"Converting {len(zips)} table(s). Free disk: {free_gb:.0f} GB\n")

    for z in zips:
        convert_one(z, out_dir, force=args.force)

    print("\nDone. Parquet files are in", out_dir)


if __name__ == "__main__":
    main()
