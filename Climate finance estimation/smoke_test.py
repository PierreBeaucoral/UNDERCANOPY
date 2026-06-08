# =============================================================================
# Climate finance estimation/smoke_test.py
# -----------------------------------------------------------------------------
# Purpose
#   Clone-runnable, LIGHT smoke test for the Python side of the UNDERCANOPY
#   replication package (the ClimateFinanceBERT pipeline under
#   "Training and Classifying/" and "Figures/graph_final.py").
#
# Why this test is deliberately light (and does NOT run the pipeline)
#   The actual pipeline needs torch + transformers + a GPU + the trained model
#   weights to run. None of that exists in CI (no GPU, no multi-GB weights, no
#   heavy deps). So we CANNOT execute the classifier here. Instead we validate
#   the two things that ARE light and clone-runnable:
#     (1) every shipped .py source compiles (syntax-level, via py_compile), and
#     (2) the shipped data artifacts are present and internally consistent.
#   This catches the most common breakages (a syntax slip on push, a corrupted
#   or out-of-sync data file) without any of the heavy machinery.
#
# Dependencies
#   Python 3 STDLIB ONLY: os, sys, json, csv, py_compile, glob.
#   No torch, no pandas, no transformers, no network, no GPU.
#
# What it checks
#   1. compile_all : every "*.py" under this directory (recursive), excluding
#                    this smoke_test.py, compiles with py_compile (doraise).
#   2. json_dicts  : dictionary_classes.json and reverse_dictionary_classes.json
#                    are non-empty and exact mutual inverses (same length).
#   3. train_set   : train_set.csv is semicolon-delimited with header
#                    text;label;relevance and has parseable data rows (HARD
#                    assertions). Label/dictionary coverage is reported as a
#                    NON-FATAL WARNING only -- see step 3 rationale below.
#
# Exit codes
#   0 -> all steps OK; prints "PY SMOKE TEST: PASS".
#   1 -> any step FAILED; prints "PY SMOKE TEST: FAIL".
#   Suitable as a CI gate.
# =============================================================================

import os
import sys
import json
import csv
import py_compile
import glob

# Resolve everything relative to THIS file so the test runs from any CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "Data")

DICT_PATH = os.path.join(DATA, "dictionary_classes.json")
REVERSE_DICT_PATH = os.path.join(DATA, "reverse_dictionary_classes.json")
TRAIN_SET_PATH = os.path.join(DATA, "train_set.csv")

# -----------------------------------------------------------------------------
# Status accumulation + step wrapper (mirrors the R smoke_test.R structure).
# -----------------------------------------------------------------------------
_step_names = []
_step_status = []  # "OK" or "FAIL"


def run_step(name, fn):
  """Run fn() inside a try/except, record OK/FAIL, and report the message."""
  try:
    fn()
    _step_names.append(name)
    _step_status.append("OK")
  except Exception as exc:  # noqa: BLE001 - we want every failure logged, not raised
    print("[FAIL] {}: {}".format(name, exc))
    _step_names.append(name)
    _step_status.append("FAIL")


# =============================================================================
# Step 1 — Every shipped .py compiles (syntax-level).
# =============================================================================
def step_compile_all():
  pattern = os.path.join(HERE, "**", "*.py")
  py_files = sorted(glob.glob(pattern, recursive=True))

  # Exclude this smoke test itself.
  self_path = os.path.abspath(__file__)
  py_files = [p for p in py_files if os.path.abspath(p) != self_path]

  if len(py_files) < 1:
    raise AssertionError("no .py files found to compile under {}".format(HERE))

  failures = []
  for path in py_files:
    try:
      py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as exc:
      failures.append("{}: {}".format(os.path.relpath(path, HERE), exc.msg))

  if failures:
    raise AssertionError(
      "compile error(s) in {} file(s):\n  - {}".format(
        len(failures), "\n  - ".join(failures)
      )
    )

  print("  compiled {} .py file(s) OK".format(len(py_files)))


# =============================================================================
# Step 2 — Forward / reverse class dictionaries are exact mutual inverses.
# =============================================================================
def step_json_dicts():
  with open(DICT_PATH, "r", encoding="utf-8") as fh:
    forward = json.load(fh)
  with open(REVERSE_DICT_PATH, "r", encoding="utf-8") as fh:
    reverse = json.load(fh)

  if not isinstance(forward, dict) or len(forward) == 0:
    raise AssertionError("dictionary_classes.json is not a non-empty dict")
  if not isinstance(reverse, dict) or len(reverse) == 0:
    raise AssertionError("reverse_dictionary_classes.json is not a non-empty dict")

  if len(forward) != len(reverse):
    raise AssertionError(
      "length mismatch: forward={}, reverse={}".format(len(forward), len(reverse))
    )

  for name, idx in forward.items():
    key = str(idx)
    if key not in reverse:
      raise AssertionError(
        "forward id {!r} (class {!r}) missing as key in reverse dict".format(idx, name)
      )
    if reverse[key] != name:
      raise AssertionError(
        "inverse mismatch at id {!r}: reverse={!r} != forward name {!r}".format(
          idx, reverse[key], name
        )
      )

  print("  forward/reverse dicts are exact inverses; {} classes".format(len(forward)))


# =============================================================================
# Step 3 — train_set.csv structure (HARD) + label/dict coverage (WARNING ONLY).
#
# Self-contained: reloads the forward dict inside its own try so a step-2
# failure does not cascade into a confusing step-3 error.
#
# WHY label-coverage is a WARNING, not a failure:
#   multi-classifier.py builds its label dictionary DYNAMICALLY at train time
#   from df[df.relevance==1]['label'].unique() and then WRITES out
#   dictionary_classes.json / reverse_dictionary_classes.json from that. So the
#   shipped dictionary_classes.json is an OUTPUT ARTIFACT, not a training INPUT:
#   the pipeline regenerates it from train_set.csv and never consumes the shipped
#   copy. A drift between the shipped dict and the shipped train_set labels does
#   NOT break replication -- it is a stale-artifact diagnostic. We therefore
#   surface it for maintainer attention but do not block the suite.
#
# HARD assertions (step FAILs -> exit 1):
#   file opens, header is exactly ["text","label","relevance"], >0 data rows,
#   relevance values parse as float.
# =============================================================================
def step_train_set():
  with open(DICT_PATH, "r", encoding="utf-8") as fh:
    forward = json.load(fh)
  valid_labels = set(forward.keys())

  expected_header = ["text", "label", "relevance"]
  n_total = 0
  n_pos = 0
  n_neg = 0
  missing_labels = set()       # relevance==1 labels absent from the shipped dict
  seen_pos_labels = set()      # all relevance==1 labels actually observed

  # HARD: file must open.
  with open(TRAIN_SET_PATH, "r", encoding="utf-8", newline="") as fh:
    reader = csv.reader(fh, delimiter=";")
    try:
      header = next(reader)
    except StopIteration:
      raise AssertionError("train_set.csv is empty (no header)")

    # HARD: header must be exactly the expected three columns.
    if header != expected_header:
      raise AssertionError(
        "unexpected header: {!r} (expected {!r})".format(header, expected_header)
      )

    for row in reader:
      if len(row) < 3:
        # Skip blank trailing lines; flag genuinely malformed rows.
        if not any(cell.strip() for cell in row):
          continue
        raise AssertionError("malformed row with < 3 fields: {!r}".format(row))

      n_total += 1
      label = row[1]
      # HARD: relevance must parse as float.
      try:
        relevance = float(row[2])
      except ValueError:
        raise AssertionError(
          "non-numeric relevance {!r} in row {!r}".format(row[2], row)
        )

      if relevance == 1.0:
        n_pos += 1
        seen_pos_labels.add(label)
        if label not in valid_labels:
          missing_labels.add(label)
      elif relevance == 0.0:
        n_neg += 1
      else:
        raise AssertionError("unexpected relevance value {!r}".format(relevance))

  # HARD: there must be at least one data row.
  if n_total < 1:
    raise AssertionError("train_set.csv has no data rows")

  # Always report the row counts.
  print(
    "  train_set.csv OK: {} rows ({} relevance==1, {} relevance==0)".format(
      n_total, n_pos, n_neg
    )
  )

  # NON-FATAL: label/dict coverage drift. multi-classifier.py rebuilds the dict
  # from train_set.csv (df['label'].unique()), so the shipped dict is an output
  # artifact, not a training input; this drift does not block replication.
  if missing_labels:
    sample = sorted(missing_labels)[:10]
    print(
      "  [WARN] train_set: {} relevance==1 label(s) not found in "
      "dictionary_classes.json (dict is regenerated by multi-classifier.py from "
      "train_set, so this is non-fatal drift): {}".format(
        len(missing_labels), ", ".join(repr(s) for s in sample)
      )
    )

  unused_keys = valid_labels - seen_pos_labels
  if unused_keys:
    sample = sorted(unused_keys)[:10]
    print(
      "  [WARN] train_set: {} dictionary_classes.json key(s) never appear as a "
      "relevance==1 label: {}".format(
        len(unused_keys), ", ".join(repr(s) for s in sample)
      )
    )


# =============================================================================
# Run all steps, print summary table, exit with appropriate status.
# =============================================================================
def main():
  run_step("compile_all", step_compile_all)
  run_step("json_dicts", step_json_dicts)
  run_step("train_set", step_train_set)

  bar = "=" * 50
  print("\n" + bar)
  print("PYTHON REPLICATION SMOKE TEST - SUMMARY")
  print(bar)
  width = max(len(n) for n in _step_names)
  for name, status in zip(_step_names, _step_status):
    print("  {:<{w}}  {}".format(name, status, w=width))
  print(bar)

  if "FAIL" in _step_status:
    print("PY SMOKE TEST: FAIL")
    sys.exit(1)
  else:
    print("PY SMOKE TEST: PASS")
    sys.exit(0)


if __name__ == "__main__":
  main()
