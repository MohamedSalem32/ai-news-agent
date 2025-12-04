
from __future__ import annotations
import json, os, math
from typing import Dict, Optional
from datetime import date
from .retriever import parse_date, days_old, TODAY_FIXED

def load_credibility(data_dir: str) -> Dict[str, float]:
    p = os.path.join(data_dir, "credibility_prior.json")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    prior = obj.get("prior", {})
    default = obj.get("default", 0.5)
    # store default under special key
    prior["_default"] = float(default)
    return prior

def recency_score(date_str: str) -> float:
    """
    recency = exp(-days_old/4), with TODAY_FIXED = 2020-12-31
    If date missing, return small but non-zero value.
    """
    d = parse_date(date_str)
    if d is None:
        return math.exp(-365/4)  # treat as old
    do = days_old(d) or 0
    return math.exp(-do / 4.0)

def credibility_score(source: str, priors: Dict[str, float]) -> float:
    return float(priors.get(source, priors.get("_default", 0.5)))

def focus_score(relevance: float, recency: float, credibility: float,
                alpha: float = 0.60, beta: float = 0.30, gamma: float = 0.10) -> float:
    return alpha*float(relevance) + beta*float(recency) + gamma*float(credibility)
