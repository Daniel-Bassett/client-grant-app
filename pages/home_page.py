from dotenv import load_dotenv
import os
import glob

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import openai

from src.openai_functions import InsightGrant

st.write('Home Page')

######## SESSION STATES ########
if 'abstract' not in st.session_state:
    st.session_state['abstract'] = ''


######### LOAD KEYS & CLIENTS #########
# load_dotenv()
# openai_key = os.getenv('OPENAI_KEY')
# openai_client = openai.OpenAI(api_key=openai_key)
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
        st.session_state['results'] = grants_with_scores.sort_values(by='cosine_scores', ascending=False)
        # st.session_state.grants['cosine_scores'] = cosine_scores
        # st.write(st.session_state.grants)

@st.fragment
def get_abstract():
    st.text_area('Enter Abstract', placeholder='Press Ctrl+Enter to Apply', max_chars=3000, height= 350, key='abstract')


######## LOAD DATA ########
# df = load_data('data/hhs.parquet')
# st.session_state['grants'] = load_data('data/hhs.parquet')
st.session_state['grants'] = pd.concat([load_data(path) for path in glob.glob('data/grants/**/*.parquet', recursive=True)])


######## SHOW DATAFRAME ########
# st.data_editor(df)
# st.write(df.iloc[0]['description'])


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

if 'results' in st.session_state:
    st.dataframe(st.session_state.results.query('cosine_scores >= 0.80').drop(columns=['embeddings', 'scraped_at']), hide_index=True)
    # for result in st.session_state.embedding:
    #     st.divider()
    #     st.write(embedding)

# st.write(st.session_state.grants.drop(columns='embeddings')['cosine_scores'])