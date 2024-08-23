import pandas as pd
from datetime import datetime
from utils import db_utils

class ETLPipeline:
    def __init__(self, database):
        self.database = database
        self.engine, self.conn = db_utils.connect_to_database(self.database)
    
    def extract_data(self):
        raise NotImplementedError("El método 'extract_data' \
          debe ser implementado por la subclase")

    def transform_data(self, data):
        raise NotImplementedError("El método 'transform_data' \
          debe ser implementado por la subclase")

    def load_data(self, df):
        raise NotImplementedError("El método 'load_data' \
          debe ser implementado por la subclase")

    def run(self):
        try:
            data = self.extract_data()
            if data:
                transformed_data = self.transform_data(data)
                self.load_data(transformed_data)
        finally:
            db_utils.close_connection(self.engine, self.conn)
