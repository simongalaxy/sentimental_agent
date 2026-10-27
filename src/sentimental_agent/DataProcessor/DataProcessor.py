import os
import pandas as pd
from typing import List
from pprint import pformat
from pathlib import Path


from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.Utilities.Settings import settings



# for loading processed data.
def get_all_filepaths_in_folder(folderpath: str, logger:Logger) -> List[str]:
    dir_path = Path(folderpath)
    filepaths = [str(p) for p in dir_path.rglob("*") if p.is_file()]
    logger.info(f"FilePaths in folder - {folderpath}: \n%s", filepaths)

    return filepaths



# for loading original data.
def get_filePath(filename: str, folderpath: str, logger: Logger) -> str:
    """
    Get the file path of the current script.

    Returns:
        str: The file path of the current script.
    """
    filepath = os.path.join(folderpath, filename)
    logger.info(f"File path: {filepath}")
    
    return filepath

    
def load_clean_split_data_from_csv(path: str, logger: Logger) -> List[pd.DataFrame]:

  # load data from csv file.
  df = pd.read_csv(path)
  logger.info(df.info())

  # clean data.
  str_cols = ["Platform", "Product/Location_Name", "Review_Text"]
  df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

  # split data into a list of dataframes by platforms and products.
  dfs = []
  for platform in df["Platform"].unique():
    for product in df[df["Platform"] == platform]["Product/Location_Name"].unique():
      df_temp = df[(df["Platform"] == platform) & (df["Product/Location_Name"] == product)].reset_index()
      dfs.append(df_temp)

  logger.info(f"Total {len(dfs)} dataframes splitted.")

  return dfs

def save_df_to_csv(df: pd.DataFrame, platform: str, product: str, folderPath: str, status: str, logger: Logger) -> None:
    platform = df["Platform"].to_list()[0].replace(" ", "_")
    product = df["Product/Location_Name"].to_list()[0].replace(" ", "_")
    filename = f"./{folderPath}/{platform}-{product}-{status}.csv"
    df.to_csv(filename)

    logger.info(f"{filename} saved to {folderPath}")

    return