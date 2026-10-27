import os


from src.sentimental_agent.Utilities.Settings import settings
from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.DataProcessor.DataProcessor import load_clean_split_data_from_csv, get_filePath
from src.sentimental_agent.ReviewsClassifier.DuplicatesRemover import DuplicatesRemover
from src.sentimental_agent.ReviewsClassifier.SentimentalClassifier import SentimentalClassifier
from src.sentimental_agent.ReviewsClassifier.TopicsIdentifier import TopicsIdentifier
# from src.sentimental_agent.LLM_Agent.LLMAgent import LLMAgent


def main():
    """
    Main function to execute the data processing tasks.
    """
    # os.environ["HF_Token"] = settings.hf_token

    # initiate instance.
    logger = Logger(__name__).get_logger()
    deduplicator = DuplicatesRemover(logger=logger)
    classifier = SentimentalClassifier(logger=logger)
    identifier = TopicsIdentifier(logger=logger)

    # agent = LLMAgent(logger=logger)
   

    # Load and Prepare data.
    path = get_filePath(filename=settings.source_filename, folderpath=settings.source_folderpath, logger=logger)
    dfs = load_clean_split_data_from_csv(path=path, logger=logger)

    # remove duplicate and similar semantic meanings reviews.
    df_deduplicated = deduplicator.remove_duplicate_reviews(dfs=dfs)

    # get sentiments from reviews.
    df_sentiment = classifier.label_sentimental(df=df_deduplicated)

    # label categories by reviews.
    df_categorized = identifier.get_and_label_categories(df=df_sentiment)

    return


# Entry point of the script.
if __name__ == "__main__":
    main()