import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from typing import List


from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.Utilities.Settings import settings


class TopicsIdentifier:
    def __init__(self, logger: Logger):
        # 1. Configure Sub-modules for deterministic and cleaner results
        # Setting random_state ensures you get the exact same categories every run
        self.logger = logger
        self.umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, random_state=42)
        self.hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')

        # Ignore generic English stop words so they don't corrupt your category names
        self.vectorizer_model = CountVectorizer(stop_words="english")

        # 2. Choose a powerful Hugging Face embedding model
        # 'all-mpnet-base-v2' offers higher accuracy for nuanced text than 'all-MiniLM-L6-v2'
        self.embedding_model_name = settings.topic_embedding_model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # 3. Initialize BERTopic with control over the number of final categories
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            nr_topics="auto" # Dynamically merges highly similar categories together
            )

    def get_and_label_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Run the pipeline
        reviews = df["Review_Text"].to_list()
        topics, probs = self.topic_model.fit_transform(reviews)

        # 2. Reduce outliers (Optional but highly recommended)
        # This forces reviews grouped into 'Topic -1' (Noise) into their closest real category
        new_topics = self.topic_model.reduce_outliers(reviews, topics, strategy="embeddings")
        self.topic_model.update_topics(reviews, topics=new_topics)

        # 3. Map categories back to your original data
        df_results = pd.DataFrame({
            "Review": reviews,
            "Category_ID": new_topics
        })

        # Get category names
        topic_info = self.topic_model.get_topic_info()
        category_mapping = dict(zip(topic_info['Topic'], topic_info['Name']))
        df_results['Category_Name'] = df_results['Category_ID'].map(category_mapping)

        # 8. View the final categorized data
        self.logger.info("\n=== DISCOVERED CATEGORIES OVERVIEW ===")
        self.logger.info(topic_info[["Topic", "Count", "Name"]])

        self.logger.info("\n=== SAMPLE MAPPED REVIEWS ===")
        self.logger.info(df_results[["Review", "Category_Name"]])

        return df_results
