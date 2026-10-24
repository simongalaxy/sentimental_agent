from ollama import Client
from typing import List
from IPython.display import display, Markdown

from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.Utilities.Settings import settings


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
            # host=self.cloud_host,
            # headers=self.cloud_header

            # local settings.
            host=self.local_host,
            headers=self.local_header
        )

    
    def _call_ollama(self, prompt: str) -> str:
        response = self.Client.chat(
            model=self.cloud_model,
            messages=[{"role": "user", "content": prompt}]
            )

        summary = response['message']['content']
        display(Markdown(summary))
        print("\n")

        return summary


    def summarize_profile(self, profile: dict) -> str:
        # Send it to a local Ollama model
        prompt = f"""
        Summarize the content:

        Content:
        Platform:
        {profile.get("Platform")}

        Product:
        {profile.get("Product")}

        Category:
        {profile.get("Category")}

        Sentiment:
        {profile.get("Sentiment")}

        Reviews:
        {profile.get("Reviews")}
        """
        summary = self._call_ollama(prompt=prompt)

        return summary


    def summarize_views(self, views: List[str]) -> str:

        prompt = f"""
        summarize the following content: {views}
        """

        summary = self._call_ollama(prompt=prompt)

        return summary


    def get_insights_from_figures(self, markdown: str) -> str:

        # Send it to a local Ollama model
        prompt = f"""
        Get insights from the following content:

        {markdown}
        """
        summary = self._call_ollama(prompt=prompt)

        return summary
