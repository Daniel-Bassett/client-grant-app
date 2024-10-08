from dotenv import load_dotenv
import os

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

def find_grants(abstract):
    if len(st.session_state.abstract) > 0:
        st.session_state['embedding'] = insight_grant.get_embedding(st.session_state.abstract)
    
    if 'embedding' in st.session_state:
        cosine_scores = np.dot(np.stack(st.session_state.grants['embeddings']), st.session_state.embedding)
        st.session_state.grants['cosine_scores'] = cosine_scores

@st.fragment
def get_abstract():
    abstract = st.text_area('Enter Abstract', placeholder='Press Ctrl+Enter to Apply', max_chars=3000, height= 350, key='abstract')


######## LOAD DATA ########
# df = load_data('data/hhs.parquet')
st.session_state['grants'] = load_data('data/hhs.parquet')


######## SHOW DATAFRAME ########
# st.data_editor(df)
# st.write(df.iloc[0]['description'])


######## Abstract Input ########

get_abstract()

buttons_container = st.container()

output_placeholder = st.container()

with buttons_container:
    find_col, del_col, _ = st.columns([4, 4, 12])

    with find_col:
        st.button('Find Grants', on_click=find_grants, args=(st.session_state.abstract))
    
    with del_col:
        st.button('Delete Abstract', on_click=clear_text)

if 'cosine_scores' in st.session_state:
    st.session_state['results'] = st.session_state.cosine_scores
    st.dataframe(, hide_index=False)
    # for result in st.session_state.embedding:
    #     st.divider()
    #     st.write(embedding)
        