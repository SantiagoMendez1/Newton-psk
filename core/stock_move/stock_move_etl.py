import pandas as pd
from api_utils.api_request import request_post_data
from setup_db import connect_to_database
from datetime import datetime
class StockMoveEtl():
    def __init__(self, api_url):
        self.api_url = api_url
    
    def extract_data(self):
        payload = {
            "model": "stock.move",
            "domain": ["origin", "ilike", "%sale%"],
            "fields_names": ["id", "product", "effective_date", "origin", "quantity"],
            "limit": 5,
            "context": {"company": 1, "user": 1}
        }
        return request_post_data(self.api_url, payload)
    
    def transform_data(self, data):
        df_stock_move = pd.DataFrame(data.json())
        df_stock_move.rename(columns={"id":"id_move"}, inplace=True)
        df_stock_move["quantity"] = df_stock_move["quantity"].astype(int)
        df_stock_move["effective_date"] = pd.to_datetime(df_stock_move["effective_date"])
        df_stock_move["origin"] = df_stock_move["origin"].str.split('.').str.get(0)
        return df_stock_move
    
    def load_data(self, df):
        engine, conn = connect_to_database("TODOREPUESTOS2")
        table_name="stock_move"
        update_date = datetime.now()
        try:
            df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
            update_date_df = pd.DataFrame({'last_update_date': [update_date]}, index=[0])
            update_date_df.to_sql(name='last_update_stock_move', con=engine, if_exists='append', index=False)
            print(f"Datos cargados exitosamente en las tablas")
        except Exception as e:
            print(f"Error al cargar datos en PostgreSQL: {e}")
    
    def run(self):
        data = self.extract_data()
        transformed_data = self.transform_data(data)
        print(transformed_data)
        self.load_data(transformed_data)
        return
