from utils.db_utils import connect_to_database
import pandas as pd

class StockMoveModel():
  def __init__(self, database):
    self.database = database
    try:
      engine, conn = connect_to_database(self.database)
      query = "SELECT * FROM stock_move_ia"
      self.data = pd.read_sql(query, engine)
    except:
      self.data = pd.DataFrame()
      
  def pre_processing(self):
        """
        Preprocess the input data for time series analysis.
        """
        try:
            # try:
            #     print(self.data.dtypes)
            #     self.data['effective_date'] = pd.to_datetime(self.data['effective_date'])
            # except KeyError as e:
            #     print(f"KeyError: {e}. Ensure 'effective_date' is a column in the data.")
            #     return None
            # except Exception as e:
            #     print(f"Error converting 'effective_date' to datetime: {e}")
            #     return None

            try:
                self.data = self.data.sort_values(by='effective_date')
                most_selled_products = self.data['product'].value_counts().reset_index()
                most_selled_products.columns = ['product', 'count']
                most_selled_products = most_selled_products[most_selled_products['count'] > 1]
                self.filter_products = list(most_selled_products['product'])
            except KeyError as e:
                print(f"KeyError: {e}. Ensure 'product' and 'quantity' are columns in the data.")
                return None
            except Exception as e:
                print(f"Error processing most sold products: {e}")
                return None

            try:
                filtered_df = self.data[self.data['product'].isin(self.filter_products)]
                pivot_df = filtered_df.pivot_table(index='effective_date', columns='product', values='quantity', aggfunc='sum', fill_value=0)
                pivot_df = pivot_df.reindex(columns=self.filter_products)
            except KeyError as e:
                print(f"KeyError: {e}. Ensure 'effective_date', 'product', and 'quantity' are columns in the data.")
                return None
            except Exception as e:
                print(f"Error creating pivot table: {e}")
                return None

            try:
                compl_date_day = pd.date_range(start=pivot_df.index.min(), end=pivot_df.index.max(), freq='D')
                df_compl_day = pd.DataFrame({'effective_date': compl_date_day})
                one_prod_df_all_date = pd.merge(df_compl_day, pivot_df, on='effective_date', how='left').fillna(0)
                one_prod_df_all_date.reset_index(drop=True)
            except Exception as e:
                print(f"Error merging and filling dates: {e}")
                return None

            try:
                df_daily = one_prod_df_all_date.groupby('effective_date').sum().reset_index()
                df_daily['effective_date'] = pd.to_datetime(df_daily['effective_date'])
                df_daily.set_index('effective_date', inplace=True)
            except Exception as e:
                print(f"Error processing daily data: {e}")
                return None

            try:
                df_weekly = df_daily.resample('W').sum()
                df_weekly['year'] = df_weekly.index.year
                df_weekly['week'] = df_weekly.index.isocalendar().week
            except Exception as e:
                print(f"Error resampling data weekly: {e}")
                return None

            try:
                split = round(len(df_weekly) * 90 / 100)
                self.train = df_weekly.iloc[:split]
                self.test = df_weekly.iloc[split:]
            except Exception as e:
                print(f"Error splitting the data into train and test sets: {e}")
                return None

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None