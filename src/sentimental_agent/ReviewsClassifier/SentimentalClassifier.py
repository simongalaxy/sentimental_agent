from transformers import pipeline
from typing import List
import pandas as pd

from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.Utilities.Settings import settings

class SentimentalClassifier:
    def __init__(self, logger: Logger):
        # sentimental classifier settings.
        self.logger = logger
        self.task = settings.sentimental_task
        self.sentimental_model = settings.sentimental_model
        self.sentimental_classifier = pipeline(self.task, model=self.sentimental_model)


    # Batch process unique reviews
    def label_sentimental(self, df: pd.DataFrame) -> pd.DataFrame:
       
        self.logger.info(f"Start label sentimental for dataframe.")
        df['Sentiment'] = [res['label'] for res in self.sentimental_classifier(df["Review_Text"].tolist(), batch_size=32)]
        self.logger.info("DataFrame with sentimental: \n%s", df.head(10))

        return df

