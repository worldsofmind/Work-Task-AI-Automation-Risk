import re, json
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        import en_core_web_sm
        _nlp = en_core_web_sm.load()
except Exception:
    _nlp = None

def load_cues(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def split_lines(text: str) -> List[str]:
    parts = re.split(r"\n|•|\u2022|\r|- ", text)
    return [normalize_text(p) for p in parts if normalize_text(p)]

# --- NEW: section-header and intro filters ---
def is_section_header(line: str) -> bool:
    l = line.strip()
    # lines ending with ':' or '?' are often headers/questions
    if l.endswith(":") or l.endswith("?"):
        return True
    # bracketed sections like [What you will do]
    if l.startswith("[") and l.endswith("]"):
        return True
    # short mostly-title-case lines
    words = l.split()
    if 2 <= len(words) <= 10:
        cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
        if cap_ratio > 0.6:
            return True
    return False

def is_marketing_or_intro(line: str) -> bool:
    l = line.lower()
    return any(k in l for k in [
        "are you passionate",
        "if this passion and vision resonate",
        "join us",
        "about the job",
        "what the role is",
        "meaningful and impactful journey"
    ])

def has_action_structure(l: str, cues: Dict[str, Any]) -> bool:
    if any(v in l.split() for v in cues["action_verbs"]):
        nouns = ["report","dashboard","email","ticket","contract","statement","invoice",
                 "pricing","spreadsheet","summary","presentation","document","request",
                 "record","dataset","analysis","plan","assessment","investigation"]
        if any(n in l for n in nouns):
            return True
    return False

def is_task_like(line: str, cues: Dict[str, Any]) -> bool:
    # --- new pre-filters ---
    if is_section_header(line) or is_marketing_or_intro(line):
        return False

    l = line.lower()
    # reject if line is mostly just job requirements or qualifications
    if any(q in l for q in cues["qualification_cues"]) and not has_action_structure(l, cues):
        return False

    # core logic: needs a strong action verb or standard task cue
    if any(rc in l for rc in cues["responsibility_cues"]) and has_action_structure(l, cues):
        return True
    first_word = l.split()[0] if l.split() else ""
    if first_word in cues["action_verbs"]:
        return True
    if re.match(r"^(prepare|generate|compile|reconcile|triage|draft|maintain|lead|conduct|monitor|report|design|develop|implement|coordinate|negotiate)\b", l):
        return True
    return has_action_structure(l, cues)

def vo_phrases_spacy(text: str) -> List[str]:
    if _nlp is None:
        return []
    phrases = []
    for sent in _nlp(text).sents:
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root is None or root.pos_ != "VERB":
            continue
        dobj = [c for c in root.children if c.dep_ in ("dobj","attr")]
        pobj = [gc for c in root.children if c.dep_ == "prep"
                for gc in c.children if gc.dep_ == "pobj"]
        target = None
        if dobj:
            target = " ".join(w.text for w in sorted(dobj[0].subtree, key=lambda w: w.i))
        elif pobj:
            target = " ".join(w.text for w in sorted(pobj[0].subtree, key=lambda w: w.i))
        if target:
            advmods = [c.text for c in root.children if c.dep_ in ("advmod","npadvmod")]
            mods = " ".join(advmods) + " " if advmods else ""
            phrases.append(normalize_text(f"{root.lemma_} {mods}{target}"))
    return phrases

def canonicalize_task(line: str) -> str:
    l = line.strip().rstrip(".")
    l = re.sub(r"^(?:you will|will|responsible for|to )\s*", "", l, flags=re.I)
    l = re.split(r"[;•]|  ", l)[0]
    return normalize_text(l)

def extract_tasks(text: str, cues: Dict[str, Any]) -> List[str]:
    tasks = []
    for line in split_lines(text):
        if len(line) < 5:
            continue
        if not is_task_like(line, cues):
            continue
        vo = vo_phrases_spacy(line)
        if vo:
            for p in vo:
                if p.lower() not in {x.lower() for x in tasks}:
                    tasks.append(p)
        else:
            t = canonicalize_task(line)
            if t and t.lower() not in {x.lower() for x in tasks}:
                tasks.append(t)
    return tasks

# --- scoring & SSG match functions (unchanged) ---
def contains_any(text: str, keywords: List[str]) -> bool:
    l = text.lower()
    return any(k in l for k in keywords)

def count_hits(text: str, keywords: List[str]) -> int:
    l = text.lower()
    return sum(1 for k in keywords if k in l)

def score_routine(task: str, cues: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    l = task.lower()
    repeat = min(1.0, 0.2*count_hits(l, cues["routine_cues"]))
    predict = min(1.0, 0.25*count_hits(l, cues["predictable_cues"]))
    codify = 1.0 if contains_any(l, cues["codify_cues"]) else 0.0
    low_comm = 1.0 if any(v in l for v in cues["low_comm_verbs"]) else 0.0
    low_context = 1.0 if contains_any(l, cues["low_context_domains"]) else 0.0
    score = 0.25*repeat + 0.2*predict + 0.2*codify + 0.2*low_comm + 0.15*low_context
    why = {
        "routine_cues_matched": [k for k in cues["routine_cues"] if k in l],
        "predictable_cues_matched": [k for k in cues["predictable_cues"] if k in l],
        "codify": codify,
        "low_comm": low_comm,
        "low_context": low_context,
    }
    return round(float(score), 2), why

def score_ai_replaceability(task: str, cues: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    l = task.lower()
    digital = 1.0 if contains_any(l, cues["low_context_domains"] + ["email","csv","pdf","log","system","crm","erp","ticket"]) else 0.5
    tasktype = 0.8 if contains_any(l, sum(cues["ai_patterns"].values(), [])) else 0.0
    creativity_low = 1.0 if contains_any(l, ["reconcile","validate","compare","compile","prepare","summarize","classify","extract","triage"]) else 0.4
    social_low = 1.0 if not contains_any(l, ["negotiate","influence","persuade","facilitate","workshop","brief leadership","stakeholder"]) else 0.2
    reg_risk_low = 0.8 if not contains_any(l, ["contract","legal","privacy","pii","personal data","regulatory"]) else 0.4
    error_tolerance = 0.6 if contains_any(l, ["report","summary","draft"]) else 0.4
    maturity = 0.0
    for p, kws in cues["ai_patterns"].items():
        if any(k in l for k in kws):
            maturity = max(maturity, {"classification":0.9,"extraction":0.8,
                                      "summarisation":0.8,"reconciliation":0.7,
                                      "data_prep":0.85,"reporting":0.8}[p])
    detail = {
        "digital": round(digital, 2),
        "tasktype": round(tasktype, 2),
        "creativity_low": round(creativity_low, 2),
        "social_low": round(social_low, 2),
        "regulatory_risk_low": round(reg_risk_low, 2),
        "error_tolerance": round(error_tolerance, 2),
        "tool_maturity": round(maturity, 2),
    }
    score = 0.2*digital + 0.2*tasktype + 0.15*creativity_low + 0.1*social_low + 0.15*reg_risk_low + 0.1*error_tolerance + 0.1*maturity
    return round(float(score), 2), detail

def recommend_mode(ai_score: float) -> str:
    if ai_score >= 0.70: return "Autonomous with QA"
    if ai_score >= 0.40: return "Assist (Human-in-the-loop)"
    return "Assist-only"

def cluster_label(task: str) -> str:
    l = task.lower()
    if any(k in l for k in ["report","dashboard","kpi","summary","present"]): return "Reporting"
    if any(k in l for k in ["ticket","support","triage","customer"]): return "Customer Support"
    if any(k in l for k in ["reconcile","invoice","statement","payment","ledger","finance","pricing"]): return "Finance Ops"
    if any(k in l for k in ["contract","supplier","procure","negotiate"]): return "Procurement"
    if any(k in l for k in ["clean","transform","dataset","sql","etl","pipeline"]): return "Data Engineering"
    return "General"

class SSGTaskIndex:
    def __init__(self, df: pd.DataFrame):
        self.df = df.dropna(subset=["task_text"]).copy()
        self.df["task_text_norm"] = self.df["task_text"].astype(str).str.strip()

    def nearest(self, query: str, sector: Optional[str] = None, role: Optional[str] = None, k: int = 3):
        # keep placeholder simple; add embeddings if needed
        return []

def enrich_with_ssg(task: str, ssg_index: Optional[SSGTaskIndex], sector_hint: str = None, role_hint: str = None):
    return {"canonical_task": None, "sector": sector_hint, "role": role_hint, "match_sim": None}

def adjust_scores_with_ssg(routine_score: float, ai_score: float, ssg_enrich: Dict[str, Any]):
    return round(routine_score, 2), round(ai_score, 2)
