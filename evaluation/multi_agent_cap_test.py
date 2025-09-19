#!/usr/bin/env python3

import argparse
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=(logging.DEBUG if verbose else logging.INFO),
        format='[%(levelname)s] %(message)s'
    )

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


DEFAULT_FILE_MAP = {
    "mid_facts": "mid_facts.json",
    "even_qa":   "even_qa.json",
    # Optional extras you might use later:
    "pedQA":     "ped_qa.json",
    "vehQA":     "veh_qa.json",
    "pedVehCap": "ped_veh_cap.json",
}

DEFAULT_RULES = {
    "internal": {  
        "caption_pedestrian": "even_qa",
        "caption_vehicle":    "even_qa",
    },
    "external": {  
        "caption_pedestrian": "mid_facts",
        "caption_vehicle":    "mid_facts",
    }
}

DEFAULT_EXTERNAL_REGEX = r"^(?i)video"  

# ---------------------------
# Indexing helpers
# ---------------------------

def index_reference(ref: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """sample -> label_id -> list_of_items"""
    idx: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for sample, items in (ref or {}).items():
        for it in items or []:
            labels = it.get("labels") or []
            if not labels:
                continue
            label_id = str(labels[0])
            idx[sample][label_id].append(it)
    return idx

def index_model_caps(model_ref: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """sample -> label_id -> {'ped': str, 'veh': str}"""
    out: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for sample, items in (model_ref or {}).items():
        for it in items or []:
            labels = it.get("labels") or []
            if not labels:
                continue
            label_id = str(labels[0])
            ped = (it.get("caption_pedestrian") or "").strip()
            veh = (it.get("caption_vehicle") or "").strip()
            out[sample][label_id] = {"ped": ped, "veh": veh}
    return out

# ---------------------------
# Model & rule resolution
# ---------------------------

def parse_cli_map(overrides: List[str]) -> Dict[str, str]:
    """
    Parse --map alias=file.json entries into {alias: filename}.
    """
    out: Dict[str, str] = {}
    for entry in overrides or []:
        if "=" not in entry:
            logging.warning(f"Ignoring malformed --map '{entry}'. Expected alias=file.json")
            continue
        alias, fname = entry.split("=", 1)
        out[alias.strip()] = fname.strip()
    return out

def build_models_and_rules(
    root_dir: Path,
    cli_map: Dict[str, str],
    config_path: Optional[Path]
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], re.Pattern]:
    
    file_map = dict(DEFAULT_FILE_MAP)
    rules = {k: dict(v) for k, v in DEFAULT_RULES.items()}
    external_regex = DEFAULT_EXTERNAL_REGEX
    models: Dict[str, str] = {}

    
    if config_path:
        cfg = load_json(config_path)
        
        if isinstance(cfg.get("models"), dict):
            for alias, p in cfg["models"].items():
                models[alias] = str(p)
        
        if isinstance(cfg.get("rules"), dict):
            for scope in ("internal", "external"):
                if scope in cfg["rules"]:
                    rules.setdefault(scope, {})
                    for seg, alias in cfg["rules"][scope].items():
                        rules[scope][seg] = alias
     
        if isinstance(cfg.get("external_regex"), str):
            external_regex = cfg["external_regex"]

    
    file_map.update(cli_map)

    for alias, fname in file_map.items():
        if alias in models:
            continue  
        if not fname:
            continue
        candidate = root_dir / fname
        if candidate.exists():
            models[alias] = str(candidate)
        else:
            logging.info(f"[info] Skipping alias '{alias}' — file not found at {candidate}")

    # Validate base
    if "mid_facts" not in models:
        raise SystemExit("Required base 'mid_facts' not found. "
                         "Place mid_facts.json in --root or provide it via --config models.mid_facts")

    # Compile regex
    try:
        ext_pat = re.compile(external_regex)
    except re.error as e:
        raise SystemExit(f"Invalid external_regex '{external_regex}': {e}")

    logging.info(f"Resolved models (alias -> path): {models}")
    logging.info(f"Resolved rules: {rules}")
    logging.info(f"External regex: {external_regex}")
    return models, rules, ext_pat

# ---------------------------
# Caption selection
# ---------------------------

def is_external_sample(sample_key: str, ext_pat: re.Pattern) -> bool:
    return bool(ext_pat.search(sample_key or ""))

def pick_caption(
    sample: str,
    label_id: str,
    seg: str,                    
    scope: str,                   
    rules: Dict[str, Dict[str, str]],
    stores: Dict[str, Dict[str, Dict[str, Dict[str, str]]]],
    base_caps: Dict[str, str],
) -> str:
    alias = (rules.get(scope, {}) or {}).get(seg, "mid_facts")
    base_val = base_caps["ped"] if seg == "caption_pedestrian" else base_caps["veh"]

    if alias == "mid_facts":
        return base_val

    model_store = stores.get(alias)
    if not model_store:
        logging.warning(f"[{sample}:{label_id}] missing alias '{alias}', fallback to base for {seg}")
        return base_val

    sample_map = model_store.get(sample)
    if not sample_map:
        logging.warning(f"[{sample}:{label_id}] alias '{alias}' has no sample, fallback to base for {seg}")
        return base_val

    caps = sample_map.get(label_id)
    if not caps:
        # if only one label in that sample, use it as a pragmatic fallback
        if len(sample_map) == 1:
            caps = next(iter(sample_map.values()))
            logging.warning(f"[{sample}:{label_id}] alias '{alias}' missing label; using only available entry for {seg}")
        else:
            logging.warning(f"[{sample}:{label_id}] alias '{alias}' missing label; fallback to base for {seg}")
            return base_val

    return (caps.get("ped") if seg == "caption_pedestrian" else caps.get("veh")) or base_val


def assemble_multi_agent(
    models: Dict[str, str],
    rules: Dict[str, Dict[str, str]],
    ext_pat: re.Pattern
) -> Dict[str, Any]:
    """
    Build final JSON by iterating the base 'mid_facts' universe and overlaying captions.
    """
    
    base_data = load_json(Path(models["mid_facts"]))
    base_idx = index_reference(base_data)

    stores: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
    for alias, apath in models.items():
        if alias == "mid_facts":
            stores[alias] = index_model_caps(base_data)
        else:
            stores[alias] = index_model_caps(load_json(Path(apath)))
        logging.info(f"Indexed alias '{alias}' from {apath}")

    result: Dict[str, Any] = {}
    total = 0
    overrides_ped = 0
    overrides_veh = 0

    for sample, label_map in base_idx.items():
        scope = "external" if is_external_sample(sample, ext_pat) else "internal"
        out_items: List[Dict[str, Any]] = []

        for label_id, base_items in label_map.items():
            for base_item in base_items:
                item = deepcopy(base_item)
                base_caps = {
                    "ped": (item.get("caption_pedestrian") or "").strip(),
                    "veh": (item.get("caption_vehicle") or "").strip(),
                }

                new_ped = pick_caption(sample, label_id, "caption_pedestrian", scope, rules, stores, base_caps)
                new_veh = pick_caption(sample, label_id, "caption_vehicle",    scope, rules, stores, base_caps)

                if new_ped != base_caps["ped"]:
                    overrides_ped += 1
                if new_veh != base_caps["veh"]:
                    overrides_veh += 1

                item["caption_pedestrian"] = new_ped
                item["caption_vehicle"]    = new_veh
                out_items.append(item)
                total += 1

        result[sample] = out_items

    logging.info(f"Total items: {total} | Overrides — ped: {overrides_ped}, veh: {overrides_veh}")
    return result

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compose multi-agent captions from .json files in a directory.")
    ap.add_argument("--root", required=True, help="Directory containing the input .json files (e.g., cap_results_test).")
    ap.add_argument("--out",  default="multi_agent_cap_test.json", help="Output JSON path.")
    ap.add_argument("--map", action="append", default=[],
                    help="Override alias to filename mapping under --root, e.g., even_qa=ped_veh_cap.json (repeatable).")
    ap.add_argument("--config", help="Optional config JSON providing models/rules/external_regex.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return ap.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    root_dir = Path(args.root)
    if not root_dir.is_dir():
        raise SystemExit(f"--root '{root_dir}' is not a directory")

    cli_map = parse_cli_map(args.map)
    config_path = Path(args.config) if args.config else None

    models, rules, ext_pat = build_models_and_rules(root_dir, cli_map, config_path)

    final_data = assemble_multi_agent(models, rules, ext_pat)
    save_json(Path(args.out), final_data)
    logging.info(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
