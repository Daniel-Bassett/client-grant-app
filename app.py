import glob
import hmac

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()

######## APP LAYOUT ########
home_page = st.Page(page='pages/home-page/home_page.py', title='Home Page')
chat_page = st.Page(page='pages/chat-page/chat_page.py', title='Chat with Grants')


pg = st.navigation({
    'Home page': [home_page],
    'Chat with Grants': [chat_page],
})

pg.run()