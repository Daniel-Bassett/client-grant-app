from dotenv import load_dotenv
import os

import pandas as pd
import numpy as np
import streamlit as st
import openai

######### LOAD KEYS & CLIENTS #########
openai_key = st.secrets['openai_key']

class InsightGrant:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=openai_key)

    def openai_analysis(self, abstract, grant_summary):
        system = "Is this abstract well-aligned with this grant?"
        text = f"company summary: {abstract}\n\nAbstract: {grant_summary}"
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                "content": system},
                {"role": "user",
                    "content": f'{text}'}
            ])
        return completion.choices[0].message.content
    
    def get_embedding(self, text, model="text-embedding-ada-002"):
        """gets embedding for text using chatgpt api"""
        return self.openai_client.embeddings.create(input=[text], model=model).data[0].embedding
