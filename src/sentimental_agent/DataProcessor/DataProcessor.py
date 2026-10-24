import os
import pandas as pd
from typing import List
from pprint import pformat
from dataclasses import asdict

from pathlib import Path


from src.sentimental_agent.Utilities.Logger import Logger
from .DataClass import Profile
from src.sentimental_agent.Utilities.Settings import settings
from src.sentimental_agent.DataProcessor.DataClass import Profile


def get_all_filepaths_in_folder(folderpath: str, logger:Logger) -> List[str]:
    dir_path = Path(folderpath)
    filepaths = [str(p) for p in dir_path.rglob("*") if p.is_file()]
    logger.info(f"FilePaths in folder - {folderpath}: \n%s", filepaths)

    return filepaths


def load_processed_data(folderpath: str, logger: Logger) -> List[Profile]:

    filepaths = get_all_filepaths_in_folder(folderpath=folderpath, logger=logger)
    profiles = []
    for i, path in enumerate(filepaths, start=1):
        
        platform = path.replace("data/Processed_files/", "").split("_")[0]
        product = path.replace(f"data/Processed_files/{platform}_", "").split(".")[0]
        temp = load_csv(filepath=path, logger=logger)
        profile = Profile(
            platform=platform,
            product=product,
            reviews=None,
            processed_reviews=temp,
            summary=None
        )
        # logger.info(f"Profile: \n%s", asdict(profile))
        logger.info(f"Platform: {platform}, Product: {product} - csv file loaded.")
        profiles.append(profile)
    

    return profiles


def get_filePath(filename: str, folderpath: str, logger: Logger) -> str:
    """
    Get the file path of the current script.

    Returns:
        str: The file path of the current script.
    """
    filepath = os.path.join(folderpath, filename)
    logger.info(f"File path: {filepath}")
    
    return filepath


def load_csv(filepath: str, logger: Logger) -> pd.DataFrame | None:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    try:
        df = pd.read_csv(filepath)
        logger.info("Data loaded successfully.")
        logger.info(f"DataFrame shape: {df.shape}")
        # logger.info(f"DataFrame preview:\n{df.head(5)}")
        logger.info("#" * 50)
        return df
    except Exception as e:
        logger.error(f"An error occurred while loading the CSV file: {e}")
        raise RuntimeError(f"An error occurred while loading the CSV file: {e}")
        return


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # data cleansing and set datatype.
    
    # 1. trim string columns
    str_columns = ["Platform", "Product/Location_Name", "Review_Text"]
    df[str_columns] = df[str_columns].apply(lambda x: x.str.strip())

    # 2. set up date datatype.
    df["Date"] = df["Date"].apply(pd.to_datetime, dayfirst=False)

    return df


def save_df_to_csv(df: pd.DataFrame, platform: str, product: str, status: str, folderpath: str, logger: Logger) -> None:
    filename = f"{platform.replace(" ", "_")}_{product.replace(" ", "_")}_{status}.csv"
    df.to_csv(os.path.join(folderpath, filename))

    return
    

def load_and_prepare_datasets(filename: str, folderpath: str, logger: Logger) -> List[Profile]:

    # Get the file path of the source CSV file.
    source_filepath = get_filePath(
        filename=filename, 
        folderpath=folderpath, 
        logger=logger
        )

    # Load the CSV file into a DataFrame.
    df_original = load_csv(
        filepath=source_filepath, 
        logger=logger
        )

    # data preprocessing.
    df = preprocess_data(df=df_original)

    # prepare datasets by platforms and products.
    profiles = []
    for platform in df["Platform"].unique():
        for product in df[df["Platform"] == platform]["Product/Location_Name"].unique().tolist():
            df_by_product = df[(df["Platform"] == platform) & (df["Product/Location_Name"] == product)]\
                            .groupby(by="Review_Text")\
                            .agg(Total_Reviews=("Review_ID", "count"), Average_Rating=("Rating", "mean"), Total_Helpful_Votes=("Helpful_Votes", "sum"))\
                            .reset_index()
            # save df as csv.
            save_df_to_csv(
                df=df_by_product, 
                platform=platform, 
                product=product, 
                folderpath=settings.tempfiles_folderpath, 
                status="original", 
                logger=logger
                )

            profile = Profile(
                platform = platform,
                product = product,
                reviews = df_by_product
            )
            logger.info(f"profile: \n%s", asdict(profile))
            profiles.append(profile)
    
    logger.info(f"total profiles created: {len(profiles)}")

    return profiles
    
