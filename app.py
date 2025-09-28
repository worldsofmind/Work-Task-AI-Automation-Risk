
import streamlit as st, pandas as pd
from app.utils import load_cues, extract_tasks, score_routine, score_ai_replaceability, recommend_mode, cluster_label, enrich_with_ssg, adjust_scores_with_ssg, SSGTaskIndex

st.set_page_config(page_title='AI Task Risk Analyzer', layout='wide')
cues = load_cues('data/cue_dicts.json')
st.title('AI Task Risk Analyzer (Option A Build)')
jd = st.text_area('Paste Job Description')
if st.button('Analyze'):
    tasks = extract_tasks(jd, cues)
    st.write(tasks)
