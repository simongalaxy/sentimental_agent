from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining
import pandas as pd
from typing import List

from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.Utilities.Settings import settings


class DuplicatesRemover:
    # 1. Load a lightweight, high-performance embedding model
    def __init__(self, logger: Logger):
        self.logger = logger
        self.model_name = settings.duplicates_embedding_model
        self.model = SentenceTransformer(self.model_name)

    def _group_semantic_similar_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Extract reviews to a list
        sentences = df["Review_Text"].tolist()

        # 2. Perform Paraphrase Mining
        # (Removed the invalid 'min_score' argument)
        pairs = paraphrase_mining(self.model, sentences)

        # 3. Group the sentences using a Union-Find (Connected Components) structure
        parent = {i: i for i in range(len(sentences))}

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])  # Path compression
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        # Merge sentences ONLY if they meet or exceed the similarity threshold
        threshold = 0.85
        for score, i, j in pairs:
            if score >= threshold:
                union(i, j)

        # 4. Map the cluster root IDs directly back to your DataFrame
        # This gives every review in df a Group ID based on its semantic cluster root
        df["Semantic_Group_ID"] = [find(i) for i in range(len(sentences))]

        # 5. Extract the groupings for visual confirmation
        groupings = {}
        for i in range(len(sentences)):
            root = find(i)
            if root not in groupings:
                groupings[root] = []
            groupings[root].append(sentences[i])

        # 6. Print the Results
        self.logger.info("--- SEMANTIC GROUPINGS ---")
        deduplicated_sentences = []

        for group_id, group in groupings.items():
            self.logger.info(f"\nGroup {group_id}:")
            for sentence in group:
                self.logger.info(f" - {sentence}")

            # The representative sentence of this group is the first one
            deduplicated_sentences.append(group[0])

        self.logger.info("\n--- FINAL DEDUPLICATED LIST ---")
        for sent in deduplicated_sentences:
            self.logger.info(f"✔ {sent}")

        return df

    def _consolidate_reviews_by_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        # consolidate the df by Semantic_Group_ID.
        df_grouped = df.groupby(by=["Platform", "Product/Location_Name", "Semantic_Group_ID"]).agg(Count=("Review_ID", "count"), Average_Rating=("Rating", "mean"), Helpful_Votes=("Helpful_Votes", "sum")).reset_index()

        # get the first sentences by Semantic_Group_IDs.
        dfs_first_sentence = []
        for id in df["Semantic_Group_ID"].unique():
            temp = df[df["Semantic_Group_ID"] == id][["Semantic_Group_ID", "Review_Text"]].head(1)
            dfs_first_sentence.append(temp)

        df_first_sentence = pd.concat(dfs_first_sentence)

        # merge these 2 dataframes as output.
        df_unique_review = pd.merge(df_grouped, df_first_sentence, on="Semantic_Group_ID")

        return df_unique_review


    # def _save_unique_reviews_to_csv(self, dfs: List[pd.DataFrame]) -> None:
    #     for df in dfs:
    #     platform = df["Platform"].to_list()[0].replace(" ", "_")
    #     product = df["Product/Location_Name"].to_list()[0].replace(" ", "_")
    #     filename = f"{platform}-{product}-unique_review.csv"
    #     df[["Platform", "Product/Location_Name", "Semantic_Group_ID", "Review_Text", "Count", "Average_Rating", "Helpful_Votes"]].sort_values(by="Review_Text", ascending=True).to_csv(filename)

    #     return

    def remove_duplicate_reviews(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        dfs_processed = [self._group_semantic_similar_reviews(df=df) for df in dfs]
        dfs_unique_review = [self._consolidate_reviews_by_groups(df=df) for df in dfs_processed]
        # self._save_unique_reviews_to_csv(dfs=dfs_unique_review)
        df_deduplicated = pd.concat(dfs_unique_review).reset_index()
        self.logger.info(f"Dataframe after deduplicated: {df_deduplicated.info()}\n")
        self.logger.info(f"df_deduplcated: \n%s", df_deduplicated.head(10))
        
        return df_deduplicated

