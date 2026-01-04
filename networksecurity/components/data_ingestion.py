from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


## Configuration of the Data Ingestion config
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import certifi
import pymongo
import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    ## Default constructor where we are creating the object of DataIngestionConfig and assigning to a variable name self.data_ingestion_config
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys) 
        
    def export_collection_as_dataframe(self):
        try:
            ## Fetching the database and collection name from the data_ingestion_config object
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            ## Creating MongoDB Client to connect with MongoDB Server to fetch the data
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tls=True,
                tlsAllowInvalidCertificates=True
            )
            ## Reading data from mongodb
            collection = self.mongo_client[database_name][collection_name]

            ## Converting the data from mongodb collection to pandas dataframe
            df = pd.DataFrame(list(collection.find()))

            ## Dropping the by default _id column from the dataframe as it is not required for model training
            if "_id" in df.columns:
                df = df.drop(columns=["_id"],axis=1)

            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    ## Exporting the entire data to into Raw Data Directory as data.csv file
    ## It's is not a good practice to use the mongodb directly everytime for the model training purpose
    ## We should always have a local copy of the data
    def export_data_to_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            ## Creating Folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            ## Splitting the data into train and test set
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            ## Saving the train and test data to the respective file path
            logging.info("Saving the train and test data to the respective file path")
            logging.info("Exporting train and test data")

            ## Creating folder for training and testing  data and a file name as train.csv and test.csv
            dir_path  = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info("Exported Train and Test file path.")

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        


    def initiate_data_ingestion(self):
        try:
            ## Fetching the data from mongodb and converting to pandas dataframe
            dataframe = self.export_collection_as_dataframe()
            ## Exporting the dataframe to Raw Data Directory name as feature store and file name as data.csv
            dataframe = self.export_data_to_feature_store(dataframe)
            ## Splitting the data into train and test set
            self.split_data_as_train_test(dataframe)
            ## Preparing the Data Ingestion Artifact to return the file path of train and test file
            dataingestionartifact= DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            return dataingestionartifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)