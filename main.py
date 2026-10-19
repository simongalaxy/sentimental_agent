import os


from src.sentimental_agent.Utilities.Settings import settings
from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.DataProcessor.DataProcessor import (
    get_filePath, 
    load_csv,
    get_df_by_platform_product
)
from src.sentimental_agent.Sentimental.SentimentalClassifier import SentimentalClassifier
from src.sentimental_agent.Categorizer.ViewCategorizer import ViewCategorizer


def main():
    """
    Main function to execute the data processing tasks.
    """
    # initiate instance.
    logger = Logger(__name__).get_logger()
    senti_classifer = SentimentalClassifier(logger=logger)
    categorizer = ViewCategorizer(logger=logger)

    # load data source.
    source_filename = settings.source_filename
    source_folderpath = settings.source_folderpath

    # Get the file path of the source CSV file.
    source_filepath = get_filePath(
        filename=source_filename, 
        folderpath=source_folderpath, 
        logger=logger
        )

    # Load the CSV file into a DataFrame.
    df = load_csv(
        filepath=source_filepath, 
        logger=logger
        )

    dfs = get_df_by_platform_product(
        df=df,
        temp_folder_path=settings.tempfiles_folderpath,
        logger=logger
    )

    # get the sentimental from review_text.
    for i, item in enumerate(dfs, start=1):
        platform = item["Platform"].unique()[0]
        product = item["Product/Location_Name"].unique()[0]
        logger.info(f"Item No. {i}/{len(dfs)}:")
        logger.info(f"Platform: {platform}")
        logger.info(f"Product/Location_Name: {product}")
        df_sentimental = senti_classifer.get_sentimental_from_review(df=item, review_col="Review_Text")
        df_completed = categorizer.label_views_by_categories(df=df_sentimental)
        df_completed.to_csv(f"./Processed_files/{platform.replace(" ", "_")}_{product.replace(" ", "_")}.csv")
    
    # ananlysis.




    return


# Entry point of the script.
if __name__ == "__main__":
    main()