from ollama import Client
from typing import List
# from IPython.display import display, Markdown
import pandas as pd

from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.Utilities.Settings import settings
from src.sentimental_agent.DataProcessor.DataClass import Profile


class LLMAgent:
    def __init__(self, logger: Logger): 
        self.logger = logger

        # ollama cloud settings.
        self.cloud_host = settings.cloud_host
        self.cloud_model = settings.cloud_model
        self.cloud_api_key = settings.cloud_api_key
        self.cloud_header = {'Authorization': self.cloud_api_key}

        # ollama localhost settings.
        self.local_model = settings.local_model
        self.local_host = settings.local_host
        self.local_header = {'x-some-header': 'some-value'}

        # declare Async Client.
        self.Client = Client(
            # cloud settings.
            host=self.cloud_host,
            headers=self.cloud_header

            # local settings.
            # host=self.local_host,
            # headers=self.local_header
        )

    
    def _call_ollama(self, prompt: str) -> str:
        response = self.Client.chat(
            model=self.cloud_model,
            # model=self.local_model,
            messages=[{"role": "user", "content": prompt}]
            )

        summary = response['message']['content']

        return summary


    def summarize_views(self, profile: Profile) -> str:

        prompt = f"""
        Generate an overview on the content, summarize the content by categories and sentiments and then gain insights from the reviews for the product called {profile.product} under the platform {profile.platform}. 
        
        Notes:
        1. separate paragraphs should be provided for the overview, summaries for all categories by sentiments and insights should be provided in seperate paragraphs.
        2. all the contents should be supported by figures.

        Reviews:
        {profile.processed_reviews.to_markdown(index=False)}

        """

        summary = self._call_ollama(prompt=prompt)

        return summary


