#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patch answers in test_predictions/mid_frames_all/veh_augmented.jsonl using best_answers.json,
resolving each "best model" by **directory name inside test_predictions**.

"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from submission.postprocess_test import (
    generate_vehicle_augmented_jsonl,
    fill_test_vqa,
    generate_submission,
)


# ---------------- Types ----------------
Key = Tuple[str, Optional[str]]             # (vqa_id, label_or_None)
VLQ = Tuple[str, Optional[str], str]        # (vqa_id, label_or_None, q_norm)
SLQ = Tuple[str, Optional[str], str]        # (sample, label_or_None, q_norm)

# -------------- Small helpers ---------------
def norm_question(q: str) -> str:
   
    return (q or "").strip().lower()

def as_iter(x):

    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def normalize_key(s: str) -> str:
    
    s = s.lower()
    parts = re.split(r'[^a-z0-9]+', s)
    parts = [p for p in parts if p]
    return "_".join(parts)

def derive_sample_from_vid(vid: str) -> str:
    
    name = Path(vid).stem
    return re.sub(r"(_vehicle_view|_overhead_view|_veh|_ped)$", "", name)

def is_external_bdd(sample: Optional[str]) -> bool:
    
    return bool(sample) and str(sample).lower().startswith("video")

def same_file(a: Path, b: Path) -> bool:
   
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return a.name.lower() == b.name.lower()

# -------------- Readers / extractors ---------------
def read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def read_json_any(p: Path) -> Iterable[Dict[str, Any]]:
    if p.suffix.lower() == ".jsonl":
        yield from read_jsonl(p)
        return
    obj = json.loads(p.read_text())
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
        return
    if isinstance(obj, dict):
        for k in ("results", "data", "entries", "samples"):
            if isinstance(obj.get(k), list):
                for x in obj[k]:
                    if isinstance(x, dict):
                        yield x
                return
        yield obj
        return
    raise ValueError(f"Unrecognized JSON structure in {p}")

def extract_vids(entry: Dict[str, Any]) -> List[str]:
    
    vids = entry.get("vqa_id")
    if vids is None and isinstance(entry.get("meta"), dict):
        vids = entry["meta"].get("vqa_id")
    return [str(v) for v in as_iter(vids)]



def extract_label(entry: Dict[str, Any]) -> Optional[str]:
    
    if "label" in entry:
        return entry.get("label")
    if isinstance(entry.get("meta"), dict):
        return entry["meta"].get("label")
    return None

def extract_sample(entry: Dict[str, Any], fallback_vid: Optional[str]) -> Optional[str]:
   
    meta = entry.get("meta")
    if isinstance(meta, dict) and meta.get("sample"):
        return str(meta["sample"])
    return derive_sample_from_vid(fallback_vid) if fallback_vid else None

def extract_predictions(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    
    preds = entry.get("predictions", [])
    return [p for p in preds if isinstance(p, dict)] if isinstance(preds, list) else []


def load_best_map(best_path: Path):

    rows = json.loads(best_path.read_text())
    if not isinstance(rows, list):
        raise ValueError("best reference must be a list")
    internal_map: Dict[str, str] = {}
    external_map: Dict[str, str] = {}
    question_order: List[str] = []
    seen_q: Set[str] = set()
    all_models_raw: Set[str] = set()
    for i, r in enumerate(rows):
        if not all(k in r for k in ("question", "internal", "model")):
            raise ValueError(f"best[{i}] missing question/internal/model")
        q_raw = r["question"]
        qn = norm_question(q_raw)
        if qn not in seen_q:
            question_order.append(q_raw)
            seen_q.add(qn)
        m = str(r["model"])
        if r.get("internal", False):
            internal_map[qn] = m
        else:
            external_map[qn] = m
        all_models_raw.add(m)
    return internal_map, external_map, question_order, all_models_raw

# -------------- Variant map from test_predictions ---------------
def build_variant_map(root: Path, file_name: str) -> Dict[str, Path]:
   
    if not root.exists():
        print(f"[ERROR] test_predictions dir not found: {root}", file=sys.stderr)
        sys.exit(2)

    variant_map: Dict[str, Path] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        fp = d / file_name
        if fp.exists():
            key = normalize_key(d.name)
            variant_map[key] = fp
    if not variant_map:
        print(f"[ERROR] no variants with '{file_name}' found under {root}", file=sys.stderr)
        sys.exit(2)
    return variant_map

def resolve_model_to_variant_path(raw_model: str, variant_map: Dict[str, Path]) -> Optional[Path]:
   
    raw_key = normalize_key(raw_model)
    if raw_key in variant_map:
        return variant_map[raw_key]

    p = Path(raw_model)
    cand_keys = []
    if p.parent and p.parent.name:
        cand_keys.append(normalize_key(p.parent.name))
    if p.name:
        cand_keys.append(normalize_key(p.name))
    for ck in cand_keys:
        if ck in variant_map:
            return variant_map[ck]

    candidates: List[Tuple[int, Path]] = []
    for k, path in variant_map.items():
        if k in raw_key or raw_key in k:
            score = min(len(k), len(raw_key))
            candidates.append((score, path))
    if candidates:
        candidates.sort(key=lambda t: (-t[0], len(t[1].parent.name)))
        return candidates.first()[1] if hasattr(candidates, "first") else candidates[0][1]

    return None

def index_variant_file(fp: Path):
    """
    Build indices for a prediction file:
      by_vlq[(vid,label,q_norm)] = answer
      by_slq[(sample,label,q_norm)] = answer
    """
    by_vlq: Dict[VLQ, str] = {}
    by_slq: Dict[SLQ, str] = {}
    if not fp.exists():
        return by_vlq, by_slq

    for entry in read_json_any(fp):
        vids = extract_vids(entry)
        if not vids:
            continue
        label = extract_label(entry)
        sample = extract_sample(entry, vids[0])
        for pr in extract_predictions(entry):
            qn = norm_question(pr.get("question", ""))
            ans = pr.get("answer")
            if not qn or ans in (None, ""):
                continue
            for vid in vids:
                by_vlq[(vid, label, qn)] = ans
            if sample:
                by_slq[(sample, label, qn)] = ans
    return by_vlq, by_slq


def fetch_answer_from_variant(
    cache: Dict[Path, Tuple[Dict[VLQ, str], Dict[SLQ, str]]],
    fp: Path,
    vids: List[str],
    sample: Optional[str],
    label: Optional[str],
    q_norm: str,
) -> Optional[str]:
    """Look up answer in a variant file: exact (vid,label,q) then fallback (sample,label,q)."""
    if fp not in cache:
        cache[fp] = index_variant_file(fp)
    by_vlq, by_slq = cache[fp]
    # exact view first
    for vid in vids:
        ans = by_vlq.get((vid, label, q_norm))
        if ans not in (None, ""):
            return ans
    # sample-level fallback
    if sample:
        ans = by_slq.get((sample, label, q_norm))
        if ans not in (None, ""):
            return ans
    return None


def patch_best_from_test_predictions(
    test_predictions: Path,
    base_variant: str,
    file_name: str,
    best_path: Path,
    output_path: Path,
    strict: bool,
):
    
    variant_map = build_variant_map(test_predictions, file_name)

   
    base_key = normalize_key(base_variant)
    if base_key not in variant_map:
        print(f"[ERROR] base variant '{base_variant}' not found under {test_predictions}", file=sys.stderr)
        sys.exit(2)
    base_fp = variant_map[base_key]


    base_entries = list(read_json_any(base_fp))
    if not base_entries:
        print("[ERROR] base file has no entries.", file=sys.stderr)
        sys.exit(2)

    internal_raw, external_raw, question_order, _ = load_best_map(best_path)
    index_cache: Dict[Path, Tuple[Dict[VLQ, str], Dict[SLQ, str]]] = {}

    out_count = 0
    with output_path.open("w") as fout:
        for entry in base_entries:
            vids = extract_vids(entry)
            if not vids:
                
                fout.write(json.dumps(entry) + "\n")
                out_count += 1
                continue

            label = extract_label(entry)
            sample = extract_sample(entry, vids[0])
            dataset = "external" if is_external_bdd(sample) else "internal"

            preds = extract_predictions(entry)
            new_preds: List[Dict[str, Any]] = []

            for pr in preds:
                q_raw = pr.get("question", "")
                base_ans = pr.get("answer")
                qn = norm_question(q_raw)

               
                raw_model = internal_raw.get(qn) if dataset == "internal" else external_raw.get(qn)

                if raw_model is None:
                    
                    new_preds.append({"question": q_raw, "answer": base_ans})
                    continue

                # Resolve to variant file under test_predictions
                chosen_fp = resolve_model_to_variant_path(raw_model, variant_map)
                if chosen_fp is None:
                    if strict:
                        continue
                    new_preds.append({"question": q_raw, "answer": base_ans})
                    continue

                if same_file(chosen_fp, base_fp):
                    new_preds.append({"question": q_raw, "answer": base_ans})
                    continue

                # Try to fetch replacement from chosen variant
                rep = fetch_answer_from_variant(index_cache, chosen_fp, vids, sample, label, qn)
                if rep in (None, ""):
                    
                    if strict:
                        continue
                    new_preds.append({"question": q_raw, "answer": base_ans})
                else:
                    new_preds.append({"question": q_raw, "answer": rep})

           
            patched = dict(entry)
            patched["predictions"] = new_preds
            fout.write(json.dumps(patched) + "\n")
            out_count += 1

    print(f"Wrote {out_count} entries to {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Patch mid_frames_all predictions using best_answers.json, matching best models by directory under test_predictions."
    )
    ap.add_argument("--test_predictions", required=True, help="Root dir containing model variant subdirs (each with veh_augmented.jsonl)")
    ap.add_argument("--base_variant", default="mid_frames_all", help="Subdirectory name of the base scaffold (default: mid_frames_all)")
    ap.add_argument("--file_name", default="veh_augmented.jsonl", help="Predictions filename inside each variant dir (default: veh_augmented.jsonl)")
    ap.add_argument("--best", required=True, help="Path to best_answers.json (question→model with internal flag)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--vqa_json", type=str, help="Path to test VQA JSON for filling predictions")
    ap.add_argument("--postprocess", action="store_true", help="Also generate veh_augmented, vqa_filled, and submission files")
    ap.add_argument("--strict", action="store_true", help="Drop unanswered questions instead of keeping base answer")
    args = ap.parse_args()

    test_predictions = Path(args.test_predictions)
    best_path = Path(args.best)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_best_from_test_predictions(
        test_predictions=test_predictions,
        base_variant=args.base_variant,
        file_name=args.file_name,
        best_path=best_path,
        output_path=output_path,
        strict=args.strict,
    )
    if args.postprocess:
        print("[INFO] Running postprocess steps...")
        vqa_filled_path = output_path.parent / "vqa_filled.json"
        submission_path = output_path.parent / "submission.json"

        if not args.vqa_json:
            print("[ERROR] --vqa_json must be provided when --postprocess is used.", file=sys.stderr)
            sys.exit(1)

        fill_test_vqa(args.vqa_json, output_path, vqa_filled_path)
        generate_submission(vqa_filled_path, submission_path)



if __name__ == "__main__":
    main()






