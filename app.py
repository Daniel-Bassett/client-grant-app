import glob
import hmac

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


######## APP LAYOUT ########
home_page = st.Page(page='pages/home_page.py', title='Home Page')
chat_page = st.Page(page='pages/chat-page/chat_page.py', title='Chat with Grants')


pg = st.navigation({
    'Home page': [home_page],
    'Chat with Grants': [chat_page],
})

pg.run()