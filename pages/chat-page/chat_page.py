from dotenv import load_dotenv
import os

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import openai


######### LOAD KEYS & CLIENTS #########
openai_key = st.secrets['openai_key']
openai_client = openai.OpenAI(api_key=openai_key)

######### DEFINE FUNCTION #########
@st.cache_data
def load_data(path):
    return pd.read_parquet(path)


######## LOAD DATA ########
df = load_data('data/hhs.parquet')


######## SESSION STATES ########
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['messages'].append({'role': 'assistant', 'content': 'Hi! Please enter your abstract.'})


######## SHOW DATAFRAME ########
# st.data_editor(df)
# st.write(df.iloc[0]['description'])


######## Chat ########
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('Enter your abstract', max_chars=3000):

    st.session_state['messages'].append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)