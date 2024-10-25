from dotenv import load_dotenv
import os
import glob
import asyncio

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import openai
from anthropic import Anthropic, RateLimitError, AsyncAnthropic, InternalServerError

from src.llm_functions import InsightGrant

COSINE_THRESHOLD = 0.79

st.write('Home Page')

######## SESSION STATES ########
if 'abstract' not in st.session_state:
    st.session_state['abstract'] = ''


######### LOAD KEYS & CLIENTS #########
insight_grant = InsightGrant()


######### DEFINE FUNCTION #########
@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

def clear_text():
    st.session_state['abstract'] = ""

def find_grants():
    if len(st.session_state.abstract) > 0:
        embedding = insight_grant.get_embedding(st.session_state.abstract)
        cosine_scores = np.dot(np.stack(st.session_state.grants['embeddings']), embedding)
        grants_with_scores = st.session_state.grants.copy()
        grants_with_scores['cosine_scores'] = cosine_scores
        st.session_state['results'] = (grants_with_scores
                                       .sort_values(by='cosine_scores', ascending=False)
                                       .query('cosine_scores >= @COSINE_THRESHOLD')
                                       .drop(columns=['embeddings', 'scraped_at'])
                                       .iloc[:200])
        with st.spinner('Finding matches...'):
            matches = asyncio.run(insight_grant.analyze_matches_anthropic_only(matches=st.session_state['results'], abstract=st.session_state.abstract))
            st.session_state['matches'] = matches.query('good_match.str.contains("yes", case=False)').reset_index(drop=True)

@st.fragment
def get_abstract():
    st.text_area('Enter Abstract', placeholder='Press Ctrl+Enter to Apply', max_chars=3000, height= 350, key='abstract')


######## LOAD DATA ########
st.session_state['grants'] = pd.concat([load_data(path) for path in glob.glob('data/grants/**/*.parquet', recursive=True)])


######## SHOW DATAFRAME ########
# st.data_editor(st.session_state['grants'])
# st.write(st.session_state['grants'].iloc[0]['description'])


######## Abstract Input ########

get_abstract()

buttons_container = st.container()

output_placeholder = st.container()

with buttons_container:
    find_col, del_col, _ = st.columns([2, 2, 13])

    with find_col:
        st.button('Search', on_click=find_grants)
    
    with del_col:
        st.button('Delete', on_click=clear_text)

if 'matches' in st.session_state:
    # st.write(len(st.session_state.matches))
    st.text('Curated Matches')
    st.dataframe(st.session_state.matches, hide_index=True)
    st.divider()
    st.text('Similarity Scores')
    st.dataframe(st.session_state.results, hide_index=True)