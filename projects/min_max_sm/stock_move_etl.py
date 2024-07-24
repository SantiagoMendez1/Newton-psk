import pandas as pd
from sqlalchemy import text
from utils.api_utils import request_post_data
from utils.db_utils import connect_to_database, execute_query
from datetime import datetime

class StockMoveEtl():
    def __init__(self, database):
        self.database = database
    
    def extract_data(self):
        
        URL = f"http://127.0.0.1:8010/{self.database}/search"
        
        try:
            query = "SELECT MAX(id_move) FROM stock_move_ia"
            result = execute_query(self.database, query)
            max_id_move = result.fetchone()[0]
            if max_id_move is None:
                max_id_move = 0
        except Exception as e:
            print(f"Error durante la consulta de la base de datos: {e}")
            return None
        
        payload = {
            "model": "stock.move",
            "domain": [ ("origin", "ilike", "%sale%"), 
                        ("id", ">" ,max_id_move)],
            "fields_names": ["id", "product", "effective_date", "origin", "quantity"],
            "limit": 10,
            "order": [('id', 'ASC')],
            "context": {"company": 1, "user": 1}
        }
        
        return request_post_data(URL, payload)
    
    def transform_data(self, data):
        df_stock_move = pd.DataFrame(data.json())
        df_stock_move.rename(columns={"id":"id_move"}, inplace=True)
        df_stock_move["quantity"] = df_stock_move["quantity"].astype(int)
        df_stock_move["effective_date"] = pd.to_datetime(df_stock_move["effective_date"])
        df_stock_move = df_stock_move.dropna(subset=["origin"])
        df_stock_move["origin"] = df_stock_move["origin"].str.split('.').str.get(0)
        return df_stock_move
    
    def load_data(self, df_stock_move):
        engine, conn = connect_to_database(self.database)
        table_name="stock_move_ia"
        update_date = datetime.now()
        try:
            df_stock_move.to_sql(name=table_name, con=engine, if_exists='append', index=False)
            update_date_df = pd.DataFrame({'last_update_date': [update_date]}, index=[0])
            update_date_df.to_sql(name='last_update_stock_move', con=engine, if_exists='append', index=False)
            print(f"Datos cargados exitosamente en las tablas")
        except Exception as e:
            print(f"Error al cargar datos en PostgreSQL: {e}")
        finally:
            conn.close()
    
    def run(self):
        data = self.extract_data()
        if data:
            transformed_data = self.transform_data(data)
            self.load_data(transformed_data)
