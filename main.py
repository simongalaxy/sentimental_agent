import os


from src.sentimental_agent.Utilities.Settings import settings
from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.DataProcessor.DataProcessor import load_and_prepare_datasets, save_df_to_csv
from src.sentimental_agent.Sentimental.SentimentalClassifier import SentimentalClassifier
from src.sentimental_agent.Categorizer.ViewCategorizer import ViewCategorizer
from src.sentimental_agent.LLM_Agent.LLMAgent import LLMAgent


def main():
    """
    Main function to execute the data processing tasks.
    """
    # initiate instance.
    logger = Logger(__name__).get_logger()
    senti_classifer = SentimentalClassifier(logger=logger)
    categorizer = ViewCategorizer(logger=logger)
    agent = LLMAgent(logger=logger)

    # Load and Prepare data.
    profiles = load_and_prepare_datasets(
        filename=settings.source_filename, 
        folderpath=settings.source_folderpath, 
        logger=logger
        )

    # obtain sentimental and label the categories.
    for i, profile in enumerate(profiles, start=1):
        logger.info(f"No. {i}/{len(profiles)}: Start processing profile - Platform: {profile.platform}, Product: {profile.product}")

        # obtain sentimental from reviews.
        logger.info("Obtaining sentimental from reviews.")
        df_sentimental = senti_classifer.get_sentimental_from_review(df=profile.reviews, review_col="Review_Text")
        logger.info("Sentimental from reviews finished.")
        
        # get categories from reviews.
        logger.info("Categorize")
        processed_df = categorizer.label_views_by_categories(df=df_sentimental)

        logger.info(f"Processed df: \n%s", processed_df.head(5))
        logger.info("\n")
        
        profile.processed_reviews = processed_df
        logger.info("#"*50)
        
        save_df_to_csv(
            df=processed_df, 
            platform=profile.platform, 
            product=profile.product, 
            status="Processed", 
            folderpath=settings.processed_files_folderPath, 
            logger=logger
            )
        logger.info(f"Data preparation completed.")

        # obtain summary of views.
        


        # obtain insights.


        

    

    # Analysis


    return


# Entry point of the script.
if __name__ == "__main__":
    main()