import pandas as pd
import requests
from datetime import datetime
from utils.etl_pipeline import ETLPipeline
import utils.api_utils as api_utils
import utils.db_utils as db

class HotelEtl(ETLPipeline):
    def extract_data(self):
            URL = f"http://127.0.0.1:8010/HOTELTERRA/search"
            try:
                query = "SELECT MAX(occupation_date) FROM occupancy_history"
                result = db.execute_query(self.conn, query)
                last_date = result.fetchone()[0]
                if last_date is None:
                    last_date = "0001-01-01"
            except Exception as e:
                print(f"Error durante la consulta de la base de datos: {e}")
                return None
            payload = {
                "model": "hotel_ia.occupancy_history",
                "domain": [("occupation_date", ">", last_date)],
                "fields_names": ["id",
                                "average_price", 
                                "total_pax", 
                                "revpar", 
                                "occupation_date", 
                                "occupation_percentage"],
                "limit": 5,
                "order": [('id', 'ASC')],
                "context": {"company": 1, "user": 1}
            }
            return api_utils.request_post_data(URL, payload)

    def transform_data(self, data):
        df_hotel_occupancy = pd.DataFrame(data.json())
        df_hotel_occupancy.rename(columns={"id": "id_occupancy"}, inplace=True)
        df_hotel_occupancy["occupation_date"] = pd.to_datetime(df_hotel_occupancy["occupation_date"])
        df_hotel_occupancy["month"] = df_hotel_occupancy["occupation_date"].dt.month
        df_hotel_occupancy["day"] = df_hotel_occupancy["occupation_date"].dt.day
        return df_hotel_occupancy
  

    def load_data(self, df_hotel):
        table_name = "occupancy_history"
        update_date = datetime.now()
        try:
            df_hotel.to_sql(name=table_name, 
                            con=self.engine, 
                            if_exists='append',
                            index=False)
            update_date_df = pd.DataFrame({'last_update_date': [update_date]}, 
                                            index=[0])
            update_date_df.to_sql(name='last_update_occupancy_history', 
                                  con=self.engine, 
                                  if_exists='append', 
                                  index=False)
            print("Datos cargados exitosamente en las tablas")
        except Exception as e:
            print(f"Error al cargar datos en PostgreSQL: {e}")