import os
import json
import sys
import pandas as pd

from src.logging.logger import logger
from src.exception.exception import TextClassificationException

sys.path.append(os.getcwd())

class DataLoader:
    def __init__(self,raw_data_path :str, processed_data_path:str):
        self.raw_data_path= raw_data_path
        self.processed_data_path= processed_data_path

    
    def load_json(self)-> list:
        try:
            logger.info(f"Loading raw JSON data from: {self.raw_data_path} ")

            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError (f"Raw dataset not found at {self.raw_data_path} ")
            
            try:
                df=pd.read_json(self.raw_data_path, lines=True)
                data=df.to_dict(orient="records")
                logger.info("Loaded JSONL format successfully ")
                return data

            except ValueError:
                with open(self.raw_data_path, "r", encoding="utf-8")as f:
                    data=json.load(f)
                logger.info("Loaded standard JSON format successfully. ")
                return data
        
        except Exception as e:
            logger.error("Error while loading raw JSON data ", exc_info=True)
            raise TextClassificationException (e,sys)


    def json_to_dataframe(self,data:list)-> pd.DataFrame:
        try:
            logger.info("Converting JSON data to pandas DataFrame ")
            df=pd.json_normalize(data)
            logger.info(f"Initial DataFrame shape: {df.shape}")
            logger.info(f"Initial columns {df.columns.tolist()}")

            TEXT_CANDIDATES={"text","content","headline","title","description","short_description"}
            LABEL_CANDIDATES=["category","label","class","topic"]

            text_col=None
            label_col=None

            for col in df.columns:
                if col.lower() in TEXT_CANDIDATES:
                    text_col=col
                if col.lower() in LABEL_CANDIDATES:
                    label_col=col

            if text_col is None or label_col is None:
                raise ValueError(f"Could not find suitable text/label columns in JSON "
                                 f"Found Columns: {df.columns.to_list()}")
            
            def map_categories(self, df: pd.DataFrame) -> pd.DataFrame:
   
                CATEGORY_MAP = {
                    "POLITICS": "Politics",
                    "WORLD NEWS": "World",
                    "THE WORLDPOST": "World",
                    "WORLDPOST": "World",
                    "BUSINESS": "Business",
                    "MONEY": "Business",
                    "SPORTS": "Sports",
                    "TECH": "Technology",
                    "SCIENCE": "Science",
                    "ENTERTAINMENT": "Entertainment",
                    "COMEDY": "Entertainment",
                    "STYLE & BEAUTY": "Lifestyle",
                    "STYLE": "Lifestyle",
                    "WELLNESS": "Lifestyle",
                    "TRAVEL": "Lifestyle",
                    "FOOD & DRINK": "Lifestyle",
                    "HOME & LIVING": "Lifestyle",
                    "PARENTING": "Lifestyle",
                    "PARENTS": "Lifestyle",
                    "HEALTHY LIVING": "Lifestyle",
                    "WEDDINGS": "Lifestyle"
                }

                df["category"] = df["category"].map(lambda x: CATEGORY_MAP.get(x, "Other"))
                return df

        
        except Exception as e:
            raise TextClassificationException(e,sys)
        
    
    
    def validate_dataframe(self, df:pd.DataFrame)-> pd.DataFrame:
        try:
            logger.info("validating DataFrame ")

            before= df.shape[0]
            df= df.dropna(subset=["text","category"])
            after= df.shape[0]

            logger.info(f"Dropped {before-after} rows with missing values ")

            df["text"]=df["text"].astype(str)
            df=df[df["text"].str.strip().str.len()>0]

            logger.info(f"Final DataFrame shape after validation: {df.shape}")
            logger.info(f"Unique categories: {df['category'].nunique()}")

            return df
        
        except Exception as e:
            raise TextClassificationException(e,sys)
        


    def save_dataframe(self, df:pd.DataFrame)-> None:
        try:
            os.makedirs(os.path.dirname(self.processed_data_path),exist_ok=True)
            df.to_csv(self.processed_data_path,index=False, encoding="utf-8")
            logger.info(f"Processed dataset saved to: {self.processed_data_path}")

        except Exception as e:
            logger.info("Error while saving processed dataset ",exc_info=True)
            raise TextClassificationException(e,sys)
        


    def run_ingestion(self)->pd.DataFrame:
        try:
            raw_data=self.load_json()
            df=self.json_to_dataframe(raw_data)
            df=self.validate_dataframe(df)
            df=self.map_categories(df)
            self.save_dataframe(df)
            
            logger.info("Data ingestion completed successfully ")

        except Exception as e:
            logger.error("Data ingestion pipepline failed ", exc_info=True)
            raise TextClassificationException(e,sys)
        


if __name__=="__main__":
    RAW_JSON_PATH="data/raw_dataset.json"
    PROCESSED_CSV_PATH="data/dataset.csv"

    loader=DataLoader(raw_data_path=RAW_JSON_PATH, processed_data_path=PROCESSED_CSV_PATH)

    loader.run_ingestion()