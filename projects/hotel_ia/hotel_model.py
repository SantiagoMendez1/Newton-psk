import pandas as pd
import json
import numpy as np
import random

from sklearn.preprocessing import StandardScaler

from utils import db_utils as db

class HotelModeL:
  def __init__(self, database):
    self.database = database
    self.engine, self.conn = db.connect_to_database(self.database)
    self.data = self.load_data()
    self.seed_value = 42
    np.random.seed(self.seed_value)
    random.seed(self.seed_value) 
    
    
  def load_data(self):
        if self.conn is not None:
            try:
                query = "SELECT * FROM occupancy_history"
                data = pd.read_sql(query, self.conn)
                return data
            except Exception as e:
                print(f"Error loading data: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()
          
  def split_train_data(self):
    data = self.data
    if not data.empty:
      X = self.data[["total_pax", "revpar", "month", "day"]].values
      y = self.data["occupation_percentage"].values
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)
      train_size = int(len(self.data) * 0.8)
      X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
      y_train, y_test = y[:train_size], y[train_size:]
      
      return X_train, X_test, y_train, y_test
    
  