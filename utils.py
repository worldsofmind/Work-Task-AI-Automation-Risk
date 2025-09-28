
import re, json
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

# Optional semantic models (kept optional)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _embed_model = None
    st_util = None

# spaCy dependency parser
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

def is_task_like(line: str, cues: Dict[str, Any]) -> bool:
    l = line.lower()
    if any(q in l for q in cues["qualification_cues"]):
        if not has_action_structure(l, cues):
            return False
    if any(rc in l for rc in cues["responsibility_cues"]):
        return True
    first_word = l.split()[0] if l.split() else ""
    if first_word in cues["action_verbs"]:
        return True
    if re.match(r"^(prepare|generate|compile|reconcile|triage|draft|maintain|negotiate|investigate|monitor|report|design|develop|implement)\b", l):
        return True
    return has_action_structure(l, cues)

def has_action_structure(l: str, cues: Dict[str, Any]) -> bool:
    if any(v in l.split() for v in cues["action_verbs"]):
        nouns = ["report","dashboard","email","ticket","contract","statement","invoice","pricing","spreadsheet","summary","presentation","document","request","record","dataset","analysis","plan"]
        if any(n in l for n in nouns):
            return True
    return False

def vo_phrases_spacy(text: str) -> List[str]:
    if _nlp is None:
        return []
    phrases = []
    for sent in _nlp(text).sents:
        root = None
        for t in sent:
            if t.dep_ == "ROOT":
                root = t
                break
        if root is None or root.pos_ != "VERB":
            continue
        dobj = [c for c in root.children if c.dep_ in ("dobj","attr")]
        pobj = []
        for c in root.children:
            if c.dep_ == "prep":
                pobj.extend([gc for gc in c.children if gc.dep_ == "pobj"])
        target = None
        if dobj:
            target = " ".join(w.text for w in sorted(dobj[0].subtree, key=lambda w: w.i))
        elif pobj:
            target = " ".join(w.text for w in sorted(pobj[0].subtree, key=lambda w: w.i))
        if target:
            advmods = [c.text for c in root.children if c.dep_ in ("advmod","npadvmod")]
            mods = " ".join(advmods) + " " if advmods else ""
            phrase = f"{root.lemma_} {mods}{target}"
            phrases.append(normalize_text(phrase))
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
                if p and p.lower() not in {x.lower() for x in tasks}:
                    tasks.append(p)
        else:
            t = canonicalize_task(line)
            if t and t.lower() not in {x.lower() for x in tasks}:
                tasks.append(t)
    if _embed_model and len(tasks) > 1:
        tasks = deduplicate_semantic(tasks, threshold=0.85)
    return tasks

def deduplicate_semantic(tasks: List[str], threshold: float=0.85) -> List[str]:
    emb = _embed_model.encode(tasks, convert_to_tensor=True, normalize_embeddings=True)
    sims = st_util.cos_sim(emb, emb).cpu().numpy()
    keep, used = [], set()
    for i in range(len(tasks)):
        if i in used: continue
        keep.append(tasks[i])
        for j in range(i+1, len(tasks)):
            if sims[i, j] >= threshold:
                used.add(j)
    return keep

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
    why = {"routine_cues_matched":[k for k in cues["routine_cues"] if k in l],
           "predictable_cues_matched":[k for k in cues["predictable_cues"] if k in l],
           "codify":codify,"low_comm":low_comm,"low_context":low_context}
    return round(float(score), 2), why

def evidence_tool_maturity(task: str, cues: Dict[str, Any]):
    l = task.lower()
    evid = []
    score = 0.0
    for pattern, kw_list in cues["ai_patterns"].items():
        if any(k in l for k in kw_list):
            evid.append(pattern)
    maturity_map = {"classification":0.9,"extraction":0.8,"summarisation":0.8,"reconciliation":0.7,"data_prep":0.85,"reporting":0.8}
    if evid:
        score = max(maturity_map.get(e, 0.6) for e in evid)
    return score, evid

def score_ai_replaceability(task: str, cues: Dict[str, Any]):
    l = task.lower()
    digital = 1.0 if contains_any(l, cues["low_context_domains"] + ["email","csv","pdf","log","system","crm","erp","ticket"]) else 0.5
    tasktype = 0.0
    for kw in sum(cues["ai_patterns"].values(), []):
        if kw in l:
            tasktype = 0.8
            break
    creativity_low = 1.0 if contains_any(l, ["reconcile","validate","compare","compile","prepare","summarize","classify","extract","triage"]) else 0.4
    social_low = 1.0 if not contains_any(l, ["negotiate","influence","persuade","facilitate","workshop","brief leadership","stakeholder"]) else 0.2
    reg_risk_low = 0.8 if not contains_any(l, ["contract","legal","privacy","pii","personal data","regulatory"]) else 0.4
    error_tolerance = 0.6 if contains_any(l, ["report","summary","draft"]) else 0.4
    maturity, maturity_evid = evidence_tool_maturity(l, cues)
    score = 0.2*digital + 0.2*tasktype + 0.15*creativity_low + 0.1*social_low + 0.15*reg_risk_low + 0.1*error_tolerance + 0.1*maturity
    detail = {"digital":round(digital,2),"tasktype":round(tasktype,2),"creativity_low":round(creativity_low,2),
              "social_low":round(social_low,2),"regulatory_risk_low":round(reg_risk_low,2),
              "error_tolerance":round(error_tolerance,2),"tool_maturity":round(maturity,2),
              "pattern_evidence":maturity_evid}
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
        self.model = _embed_model
        self.emb = None
        if self.model is not None and len(self.df):
            self.emb = self.model.encode(self.df["task_text_norm"].tolist(), normalize_embeddings=True)

    def nearest(self, query: str, sector: Optional[str]=None, role: Optional[str]=None, k:int=3):
        if self.model is None or self.emb is None:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        import numpy as np
        sims = (q @ self.emb.T).flatten()
        idx = np.argsort(-sims)
        rows = []
        for i in idx[:max(k,20)]:
            r = self.df.iloc[i].to_dict()
            r["similarity"] = float(sims[i])
            rows.append(r)
        if sector:
            rows = [r for r in rows if str(r.get("sector","")).lower()==sector.lower()] or rows
        if role:
            rows = [r for r in rows if str(r.get("role","")).lower()==role.lower()] or rows
        return rows[:k]

def enrich_with_ssg(task: str, ssg_index: Optional[SSGTaskIndex], sector_hint: str=None, role_hint: str=None):
    if not ssg_index:
        return {"canonical_task": None, "sector": sector_hint, "role": role_hint, "match_sim": None}
    hits = ssg_index.nearest(task, sector_hint, role_hint, k=3)
    if not hits:
        return {"canonical_task": None, "sector": sector_hint, "role": role_hint, "match_sim": None}
    top = hits[0]
    accepted = top["similarity"] >= 0.72
    return {"canonical_task": top["task_text"] if accepted else None,
            "sector": top.get("sector", sector_hint),
            "role": top.get("role", role_hint),
            "match_sim": round(top["similarity"], 3)}

def adjust_scores_with_ssg(routine_score, ai_score, ssg_enrich):
    if ssg_enrich.get("canonical_task"):
        routine_score = min(1.0, routine_score + 0.05)
        ai_score = min(1.0, ai_score + 0.05)
    return round(routine_score, 2), round(ai_score, 2)
