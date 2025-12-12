from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List
import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file safely and validate its top-level structure."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path.name}: expected top-level mapping by culture code")
    return data


def merge_rule_files(
    rules_dir: str | Path,
    patterns: Tuple[str, ...] = ("*.yaml", "*.yml"),
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load and merge all YAML files in a directory into a single dictionary.

    Returns a structure like:
        { 'DE': { 'hygiene': { 'REQUIRED': {...}, 'SUGGESTED': {...} }, ... } }
    """
    rules_dir = Path(rules_dir)
    merged: Dict[str, Dict[str, Any]] = {}
    files: List[Path] = []

    for pat in patterns:
        files.extend(sorted(rules_dir.glob(pat)))

    # Verbose output for sanity checking
    if verbose:
        if not files:
            print(f"⚠️  No YAML files found in {rules_dir.resolve()}")
        else:
            print(f"🔎 Scanning {rules_dir.resolve()} ...")
            for p in files:
                print(f"  📘 Loading rules from: {p.name}")

    for path in files:
        data = _load_yaml(path)
        for culture, categories in data.items():
            cc = merged.setdefault(culture, {})
            category = path.stem
            cc_cat = cc.setdefault(category, {})
            if not isinstance(categories, dict):
                continue
            for req_sugg, items in categories.items():
                d = cc_cat.setdefault(req_sugg, {})
                if isinstance(items, dict):
                    for rule_id, info in items.items():
                        d[rule_id] = info

    if verbose:
        print(f"✅ Loaded {len(files)} YAML file(s) from {rules_dir.resolve()}")

    return merged


def flatten_required_rules(
    merged: Dict[str, Dict[str, Any]], culture: str
) -> List[str]:
    """
    Return a flat list of all 'correct' entries under REQUIRED rules for a culture.
    Used to compute cultural success priors.
    """
    out: List[str] = []
    culture_data = merged.get(culture, {})
    for _, cat in culture_data.items():
        req = cat.get("REQUIRED", {})
        if isinstance(req, dict):
            for rule_id, info in req.items():
                if isinstance(info, dict):
                    corr = info.get("correct", [])
                    if isinstance(corr, list):
                        out.extend([str(x).strip().lower() for x in corr])
    return out