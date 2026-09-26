#!/usr/bin/env python3
"""Build a Tibetan label.txt file for function_script/syllable/tibetan_syllable_mask.py.

The mask script looks up transcripts by an utt_id derived from metadata.csv:
parent directory name + "_" + wav stem.  This helper reads the *-uni.csv
transcript tables, matches them to metadata speech_path basenames, and writes
the exact key shape expected by the mask script.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate label.txt from transcripts/*-uni.csv for Tibetan syllable-role masks."
    )
    parser.add_argument("--metadata_file", required=True, help="metadata.csv for the new split/data.")
    parser.add_argument("--transcript_dir", default="transcripts", help="Directory containing *-uni.csv files.")
    parser.add_argument(
        "--transcript_files",
        nargs="*",
        default=None,
        help="Optional explicit transcript CSV files. Defaults to <transcript_dir>/*-uni.csv.",
    )
    parser.add_argument("--output_file", required=True, help="Output label.txt path.")
    parser.add_argument(
        "--report_file",
        default="",
        help="Optional report path. Defaults to <output_file>.report.md.",
    )
    parser.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Exit with non-zero status when any metadata row has no transcript.",
    )
    parser.add_argument(
        "--allow_duplicate_keys",
        action="store_true",
        help="Allow duplicate generated utt_id keys if their text is identical.",
    )
    return parser.parse_args()


def normalize_path_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().strip('"').replace("\\", "/")


def path_stem(value):
    text = normalize_path_text(value)
    if not text:
        return ""
    return Path(text).stem


def utt_id_from_speech_path(value):
    text = normalize_path_text(value)
    if not text:
        return ""
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[-2]}_{Path(parts[-1]).stem}"
    return Path(text).stem


def read_table(path):
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "gb18030"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except UnicodeError as exc:
            last_error = exc
    raise UnicodeError(f"Could not decode {path}: {last_error}")


def resolve_transcript_files(args):
    if args.transcript_files:
        files = [Path(item) for item in args.transcript_files]
    else:
        files = sorted(Path(args.transcript_dir).glob("*-uni.csv"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Transcript file(s) not found: " + ", ".join(missing))
    if not files:
        raise FileNotFoundError(f"No *-uni.csv files found in {args.transcript_dir}")
    return files


def find_column(columns, candidates):
    lowered = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_transcripts(files):
    by_stem = {}
    duplicate_rows = []
    empty_sentence = 0
    total_rows = 0

    for csv_path in files:
        table = read_table(csv_path)
        path_col = find_column(table.columns, {"path", "wav", "wav_path", "speech_path", "file"})
        text_col = find_column(table.columns, {"sentence", "text", "transcript", "transcription"})
        if path_col is None or text_col is None:
            raise ValueError(
                f"{csv_path} must contain path/sentence columns; got {list(table.columns)}"
            )

        for row_index, row in table.iterrows():
            total_rows += 1
            stem = path_stem(row[path_col])
            text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
            if not stem:
                continue
            if not text:
                empty_sentence += 1
                continue
            if stem in by_stem and by_stem[stem]["text"] != text:
                duplicate_rows.append(
                    {
                        "stem": stem,
                        "old_file": by_stem[stem]["file"],
                        "new_file": str(csv_path),
                        "new_row": int(row_index),
                    }
                )
                continue
            by_stem.setdefault(stem, {"text": text, "file": str(csv_path), "row": int(row_index)})

    return by_stem, {
        "total_rows": total_rows,
        "unique_stems": len(by_stem),
        "duplicate_conflicts": duplicate_rows,
        "empty_sentence": empty_sentence,
    }


def first_existing_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def build_labels(metadata_file, transcript_by_stem, allow_duplicate_keys):
    metadata = pd.read_csv(metadata_file)
    speech_col = first_existing_column(
        metadata.columns,
        ["speech_path", "clean_input", "clean_path", "source_path", "wav_path", "path"],
    )
    if speech_col is None:
        raise ValueError(
            f"{metadata_file} must contain one of: "
            "speech_path, clean_input, clean_path, source_path, wav_path, path."
        )

    labels = {}
    matched_rows = 0
    duplicate_same_text = 0
    missing = []
    duplicate_keys = []

    for row_index, row in metadata.iterrows():
        speech_path = row[speech_col]
        stem = path_stem(speech_path)
        utt_id = utt_id_from_speech_path(speech_path)
        if not stem or not utt_id:
            missing.append({"row": int(row_index), "stem": stem, "utt_id": utt_id, "speech_path": speech_path})
            continue

        transcript = transcript_by_stem.get(stem)
        if transcript is None:
            missing.append({"row": int(row_index), "stem": stem, "utt_id": utt_id, "speech_path": speech_path})
            continue

        text = transcript["text"]
        matched_rows += 1
        if utt_id in labels:
            if labels[utt_id] != text:
                duplicate_keys.append({"row": int(row_index), "utt_id": utt_id, "speech_path": speech_path})
            else:
                duplicate_same_text += 1
            continue

        labels[utt_id] = text

    return labels, {
        "metadata_rows": len(metadata),
        "speech_column": speech_col,
        "matched_rows": matched_rows,
        "unique_labels": len(labels),
        "duplicate_same_text": duplicate_same_text,
        "missing_rows": missing,
        "duplicate_keys": duplicate_keys,
    }


def write_label_file(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for utt_id in sorted(labels):
            handle.write(f"{utt_id}\t{labels[utt_id]}\n")


def write_report(path, args, transcript_files, transcript_stats, label_stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tibetan label build report",
        "",
        f"- metadata_file: `{args.metadata_file}`",
        f"- output_file: `{args.output_file}`",
        f"- transcript_files: {len(transcript_files)}",
        f"- transcript_rows: {transcript_stats['total_rows']}",
        f"- transcript_unique_stems: {transcript_stats['unique_stems']}",
        f"- transcript_empty_sentence: {transcript_stats['empty_sentence']}",
        f"- transcript_duplicate_conflicts: {len(transcript_stats['duplicate_conflicts'])}",
        f"- metadata_rows: {label_stats['metadata_rows']}",
        f"- metadata_speech_column: `{label_stats['speech_column']}`",
        f"- matched_rows: {label_stats['matched_rows']}",
        f"- unique_labels: {label_stats['unique_labels']}",
        f"- duplicate_same_text_rows: {label_stats['duplicate_same_text']}",
        f"- missing_rows: {len(label_stats['missing_rows'])}",
        f"- duplicate_generated_utt_id: {len(label_stats['duplicate_keys'])}",
        "",
    ]

    if label_stats["missing_rows"]:
        lines.append("## First missing rows")
        for item in label_stats["missing_rows"][:30]:
            lines.append(
                f"- row={item['row']}, stem=`{item['stem']}`, utt_id=`{item['utt_id']}`, "
                f"speech_path=`{item['speech_path']}`"
            )
        lines.append("")

    if transcript_stats["duplicate_conflicts"]:
        lines.append("## First transcript duplicate conflicts")
        for item in transcript_stats["duplicate_conflicts"][:30]:
            lines.append(
                f"- stem=`{item['stem']}`, old_file=`{item['old_file']}`, "
                f"new_file=`{item['new_file']}`, new_row={item['new_row']}"
            )
        lines.append("")

    if label_stats["duplicate_keys"]:
        lines.append("## First duplicate generated utt_id rows")
        for item in label_stats["duplicate_keys"][:30]:
            lines.append(
                f"- row={item['row']}, utt_id=`{item['utt_id']}`, speech_path=`{item['speech_path']}`"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    transcript_files = resolve_transcript_files(args)
    transcript_by_stem, transcript_stats = load_transcripts(transcript_files)
    labels, label_stats = build_labels(
        args.metadata_file,
        transcript_by_stem,
        allow_duplicate_keys=args.allow_duplicate_keys,
    )

    if label_stats["duplicate_keys"]:
        print(
            f"ERROR: {len(label_stats['duplicate_keys'])} duplicate generated utt_id keys found. "
            "Use --allow_duplicate_keys only if identical duplicates are expected.",
            file=sys.stderr,
        )
        return 2

    write_label_file(args.output_file, labels)
    report_file = args.report_file or f"{args.output_file}.report.md"
    write_report(report_file, args, transcript_files, transcript_stats, label_stats)

    print(f"Wrote {len(labels)} labels to {args.output_file}")
    print(f"Wrote report to {report_file}")
    print(
        f"Matched {label_stats['matched_rows']}/{label_stats['metadata_rows']} metadata rows; "
        f"missing={len(label_stats['missing_rows'])}"
    )

    if args.fail_on_missing and label_stats["missing_rows"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
