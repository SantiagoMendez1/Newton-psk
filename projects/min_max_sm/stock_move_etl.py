import pandas as pd
# from sqlalchemy import text
from utils.api_utils import request_post_data
import utils.db_utils as db
from datetime import datetime

class StockMoveEtl():
    def __init__(self, database):
        self.database = database
        self.engine, self.conn = db.connect_to_database(self.database)
    
    def extract_data(self):
        URL = f"http://127.0.0.1:8010/TODOREPUESTOS/search"
        try:
            query = "SELECT MAX(id_move) FROM stock_move_ia"
            result = db.execute_query(self.conn, query)
            max_id_move = result.fetchone()[0]
            if max_id_move is None:
                max_id_move = 0
        except Exception as e:
            print(f"Error durante la consulta de la base de datos: {e}")
            return None
        payload = {
                    "model": "stock.move",
                    "domain": [("origin", "ilike", "%sale%"), ("id", ">", max_id_move)],
                    "fields_names": ["id", "product", "effective_date", "origin", "quantity"],
                    "limit": 5000,
                    "order": [('id', 'ASC')],
                    "context": {"company": 1, "user": 1}
                }
        
        return request_post_data(URL, payload)
    
    def transform_data(self, data):
        df_stock_move = pd.DataFrame(data.json())
        df_stock_move.rename(columns={"id": "id_move"}, inplace=True)
        df_stock_move["quantity"] = df_stock_move["quantity"].astype(int)
        df_stock_move["effective_date"] = pd.to_datetime(df_stock_move["effective_date"])
        df_stock_move = df_stock_move.dropna(subset=["origin"])
        df_stock_move["origin"] = df_stock_move["origin"].str.split('.').str.get(0)
        return df_stock_move
    
    def load_data(self, df_stock_move):
        table_name = "stock_move_ia"
        update_date = datetime.now()
        try:
            df_stock_move.to_sql(name=table_name, con=self.engine, if_exists='append', index=False)
            update_date_df = pd.DataFrame({'last_update_date': [update_date]}, index=[0])
            update_date_df.to_sql(name='last_update_stock_move', con=self.engine, if_exists='append', index=False)
            print("Datos cargados exitosamente en las tablas")
        except Exception as e:
            print(f"Error al cargar datos en PostgreSQL: {e}")
    
    def run(self):
        try:
            data = self.extract_data()
            if data:
                transformed_data = self.transform_data(data)
                self.load_data(transformed_data)
        finally:
            db.close_connection(self.engine, self.conn)
    