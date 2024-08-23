import pandas as pd
import json
import numpy as np
import random

import utils.db_utils as db
import utils.api_utils as api
import tools

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV

import warnings
warnings.filterwarnings("ignore")

class StockMoveModel:
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
                query = "SELECT * FROM stock_move_ia"
                data = pd.read_sql(query, self.conn)
                return data
            except Exception as e:
                print(f"Error loading data: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def pre_processing(self):
        """
        Preprocess the input data for time series analysis.
        """
        try:
            self.data = self.data.sort_values(by='effective_date')
            most_selled_products = self.data['product'].value_counts().reset_index()
            most_selled_products.columns = ['product', 'count']
            most_selled_products = most_selled_products[most_selled_products['count'] > 10]
            self.filter_products = list(most_selled_products['product'])
            filtered_df = self.data[self.data['product'].isin(self.filter_products)]
            pivot_df = filtered_df.pivot_table(index='effective_date', 
                                                columns='product', 
                                                values='quantity', 
                                                aggfunc='sum', 
                                                fill_value=0)
            pivot_df = pivot_df.reindex(columns=self.filter_products)
            compl_date_day = pd.date_range(start=pivot_df.index.min(), 
                                            end=pivot_df.index.max(), 
                                            freq='D')
            df_compl_day = pd.DataFrame({'effective_date': compl_date_day})
            one_prod_df_all_date = pd.merge(df_compl_day, pivot_df, 
                                            on='effective_date', 
                                            how='left').fillna(0)
            one_prod_df_all_date.reset_index(drop=True)
            df_daily = one_prod_df_all_date.groupby('effective_date').sum().reset_index()
            df_daily['effective_date'] = pd.to_datetime(df_daily['effective_date'])
            df_daily.set_index('effective_date', inplace=True)
            df_weekly = df_daily.resample('W').sum()
            df_weekly['year'] = df_weekly.index.year
            df_weekly['week'] = df_weekly.index.isocalendar().week
            split = round(len(df_weekly) * 90 / 100)
            self.train = df_weekly.iloc[:split]
            self.test = df_weekly.iloc[split:]
            
        except KeyError as e:
            print(f"KeyError: {e}. Ensure 'product' and 'quantity'\
                    are columns in the data.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def model_hyperparam_tuning(self, run=False):
        """
        Perform hyperparameter tuning for a RandomForestRegressor model.
        
        Args:
        run (bool): Flag to indicate whether to perform hyperparameter tuning or not. 
        If False, attempts to load from file.
        """
        products_to_train = self.filter_products[0:2]
        if run:
            self.all_best_params = []
            for product in products_to_train:
                X_train = self.train[['week', 'year']]
                y_train = self.train[product]
                
                param_grid = {'n_estimators': np.arange(400, 2200, 200),
                                'max_depth': [10, 15, 18, 20],
                                'max_features': ['sqrt', 'log2'],
                                'min_samples_split': [2, 5, 8, 10]}

                random_forest_model = RandomForestRegressor(random_state=42)

                bayes_search = BayesSearchCV(
                                            random_forest_model,
                                            param_grid,
                                            scoring='neg_mean_squared_error',
                                            cv=2,
                                            n_iter=50,
                                            random_state=42,
                                            n_jobs=-1,
                                            )

                bayes_search.fit(X_train, y_train)
                best_params = bayes_search.best_params_
                best_model = bayes_search.best_estimator_
                items_params = {key : value for key, value in best_params.items()}
                items_params = json.dumps(items_params)
                product_trained = {
                    "id_product": product,
                    "name":self.find_product_name(product),
                    "hyperparams":items_params
                }
                self.all_best_params.append(product_trained)
            df_product_trained = pd.DataFrame(self.all_best_params)
            print(df_product_trained)
            df_product_trained.to_sql(name="products2_trained", 
                                      con=self.engine,                  
                                      if_exists='append', 
                                      index=False)
            return None
        
    def prediction(self, product, months_to_predict):
        """
        Predict future quantities for a given product using a trained RandomForestRegressor model.
        
        Args:
        product (int): The product ID for which predictions are to be made.
        months_to_predict (int): The number of months into the future for which predictions are needed.

        Returns:
        df_pred_monthly: A dataframe with the predictions of the months_to_predict.
        """
        self.product = str(product)
        self.months_to_predict = months_to_predict
        
        
        try:
            self.query = f"SELECT hyperparams from product_trained where id_product = '{self.product}'"
            self.result_query_best_params = db.execute_query(self.conn, 
                                                            self.query)
        except:
            return None
        
        best_params_str = self.result_query_best_params.fetchone()[0]
        best_params = json.loads(best_params_str)
        
        X_train = self.train[['week', 'year']]
        y_train = self.train[self.product]

        X_test = self.test[['week', 'year']]
        y_test = self.test[self.product]

        y_pred = y_test.copy()
        last_date = y_pred.index[-1]
        final_date = last_date + pd.DateOffset(months=self.months_to_predict)
        new_dates = pd.date_range(start=last_date, end=final_date, freq='W-SUN')[1:]
        df_new_dates = pd.DataFrame(index=new_dates)
        df_new_dates['week'] = df_new_dates.index.isocalendar().week
        df_new_dates['year'] = df_new_dates.index.year
        best_model = RandomForestRegressor(random_state=self.seed_value, **best_params)
        best_model.fit(X_train, y_train)
        self.y_hat_test = np.round(np.abs(best_model.predict(X_test)))
        rmse_test = np.sqrt(mean_squared_error(y_pred, self.y_hat_test))
        sigma = np.abs(y_pred - self.y_hat_test).std()
        y_pred = pd.DataFrame({'Quantity': y_pred}, index=y_test.index)
        y_pred['Quantity Prediction'] = self.y_hat_test
        self.y_pred = y_pred.resample('ME').sum()
        self.y_pred['Lower Quantity'] = np.round(self.y_pred['Quantity Prediction'] - 1.96 * sigma).astype(int)
        self.y_pred['Higher Quantity'] = np.round(self.y_pred['Quantity Prediction'] + 1.96 * sigma).astype(int)

        y_hat = np.round(np.abs(best_model.predict(df_new_dates[['week', 'year']]))).astype(int)
        df_new_dates['rf_prediction'] = y_hat
        self.df_pred_monthly = df_new_dates['rf_prediction'].resample('ME').sum()
        self.df_pred_monthly = pd.DataFrame({'Quantity Prediction':self.df_pred_monthly}, index=self.df_pred_monthly.index)
        self.df_pred_monthly.index.name = 'Date' 
        self.df_pred_monthly.index = self.df_pred_monthly.index.strftime('%Y-%m-%d')
        
        self.df_pred_monthly['Lower Quantity'] = np.round(self.df_pred_monthly['Quantity Prediction'] - 1.96 * sigma).astype(int)
        self.df_pred_monthly['Higher Quantity'] = np.round(self.df_pred_monthly['Quantity Prediction'] + 1.96 * sigma).astype(int)
        self.df_pred_monthly = self.df_pred_monthly.iloc[:-1]
        return self.df_pred_monthly
    

    def find_product_name(self, id_product):
        config = tools.load_config()
        URL_API = config.get('URL_API')
        payload = {"model": "product.product",
                    "domain": [("id", "=", id_product)],
                    "fields_names": ["name"],
                    "context": {"company": 1, "user": 1}
                }
        response = api.request_post_data(URL_API, payload)
        product = response.json().pop()
        return product["name"]

# import pandas as pd
# import json
# import numpy as np
# import random
# import joblib
# import os

# import utils.db_utils as db
# import utils.api_utils as api

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from skopt import BayesSearchCV

# import warnings
# warnings.filterwarnings("ignore")

# class StockMoveModel:
#     def __init__(self, database):
#         self.database = database
#         self.engine, self.conn = db.connect_to_database(self.database)
#         self.data = self.load_data()
#         self.seed_value = 42
#         np.random.seed(self.seed_value)
#         random.seed(self.seed_value)    

#     def load_data(self):
#         if self.conn is not None:
#             try:
#                 query = "SELECT * FROM stock_move_ia"
#                 data = pd.read_sql(query, self.conn)
#                 return data
#             except Exception as e:
#                 print(f"Error loading data: {e}")
#                 return pd.DataFrame()
#         else:
#             return pd.DataFrame()

#     def pre_processing(self):
#         """
#         Preprocess the input data for time series analysis.
#         """
#         try:
#             self.data = self.data.sort_values(by='effective_date')
#             most_selled_products = self.data['product'].value_counts().reset_index()
#             most_selled_products.columns = ['product', 'count']
#             most_selled_products = most_selled_products[most_selled_products['count'] > 10]
#             self.filter_products = list(most_selled_products['product'])
#             filtered_df = self.data[self.data['product'].isin(self.filter_products)]
#             pivot_df = filtered_df.pivot_table(index='effective_date', 
#                                                 columns='product', 
#                                                 values='quantity', 
#                                                 aggfunc='sum', 
#                                                 fill_value=0)
#             pivot_df = pivot_df.reindex(columns=self.filter_products)
#             compl_date_day = pd.date_range(start=pivot_df.index.min(), 
#                                             end=pivot_df.index.max(), 
#                                             freq='D')
#             df_compl_day = pd.DataFrame({'effective_date': compl_date_day})
#             one_prod_df_all_date = pd.merge(df_compl_day, pivot_df, 
#                                             on='effective_date', 
#                                             how='left').fillna(0)
#             one_prod_df_all_date.reset_index(drop=True)
#             df_daily = one_prod_df_all_date.groupby('effective_date').sum().reset_index()
#             df_daily['effective_date'] = pd.to_datetime(df_daily['effective_date'])
#             df_daily.set_index('effective_date', inplace=True)
#             df_weekly = df_daily.resample('W').sum()
#             df_weekly['year'] = df_weekly.index.year
#             df_weekly['week'] = df_weekly.index.isocalendar().week
#             split = round(len(df_weekly) * 90 / 100)
#             self.train = df_weekly.iloc[:split]
#             self.test = df_weekly.iloc[split:]
            
#         except KeyError as e:
#             print(f"KeyError: {e}. Ensure 'product' and 'quantity' are columns in the data.")
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")

#     def model_hyperparam_tuning(self, run=False):
#         """
#         Perform hyperparameter tuning for a RandomForestRegressor model.
        
#         Args:
#         run (bool): Flag to indicate whether to perform hyperparameter tuning or not. 
#         If False, attempts to load from file.
#         """
#         products_to_train = self.filter_products[0:2]
#         if run:
#             self.all_best_params = []
#             for product in products_to_train:
#                 X_train = self.train[['week', 'year']]
#                 y_train = self.train[product]
                
#                 param_grid = {'n_estimators': np.arange(400, 2200, 200),
#                                 'max_depth': [10, 15, 18, 20],
#                                 'max_features': ['sqrt', 'log2'],
#                                 'min_samples_split': [2, 5, 8, 10]}

#                 random_forest_model = RandomForestRegressor(random_state=42)

#                 bayes_search = BayesSearchCV(
#                                             random_forest_model,
#                                             param_grid,
#                                             scoring='neg_mean_squared_error',
#                                             cv=2,
#                                             n_iter=50,
#                                             random_state=42,
#                                             n_jobs=-1,
#                                             )

#                 bayes_search.fit(X_train, y_train)
#                 best_params = bayes_search.best_params_
#                 best_model = bayes_search.best_estimator_
#                 items_params = {key : value for key, value in best_params.items()}
#                 items_params = json.dumps(items_params)
                
#                 model_dir = 'models'
#                 if not os.path.exists(model_dir):
#                     os.makedirs(model_dir)
                
#                 model_filename = f"model_{product}.pkl"
#                 try:
#                     joblib.dump(best_model, f"models/{model_filename}")
#                     print(f"Model saved successfully as {model_filename}")
#                 except Exception as e:
#                     print(f"Error saving model: {e}")
                
#                 product_trained = {
#                     "id_product": product,
#                     "name": self.find_product_name(product),
#                     "model_filename": model_filename,
#                     "hyperparams": items_params
#                 }
#                 self.all_best_params.append(product_trained)
#             df_product_trained = pd.DataFrame(self.all_best_params)
#             print(df_product_trained)
#             df_product_trained.to_sql(name="products_trained", 
#                                       con=self.engine,                  
#                                       if_exists='append', 
#                                       index=False)
#             print('ya se hizooooooooooooo')
#             return None

#     def retrain_model(self, product):
#         """
#         Re-train the model for a given product using the latest data.

#         Args:
#         product (int): The product ID for which the model needs to be retrained.
#         """
#         self.product = str(product)
#         self.model_hyperparam_tuning(run=True)
#         print(f"Model for product {self.product} has been retrained.")

#     def prediction(self, product, months_to_predict):
#         """
#         Predict future quantities for a given product using a trained RandomForestRegressor model.
        
#         Args:
#         product (int): The product ID for which predictions are to be made.
#         months_to_predict (int): The number of months into the future for which predictions are needed.

#         Returns:
#         df_pred_monthly: A dataframe with the predictions of the months_to_predict.
#         """
#         self.product = str(product)
#         print(type(self.product))
#         self.months_to_predict = months_to_predict
        
#         try:
#             self.query = f"SELECT hyperparams, model_filename FROM product_trained WHERE id_product = '{self.product}'"
#             self.result_query_best_params = db.execute_query(self.conn, self.query)
#             best_params, model_filename = self.result_query_best_params.fetchone()
#             best_params = json.loads(best_params)
            
#             # Verifica la existencia del archivo del modelo
#             if not os.path.exists(f"{model_filename}"):
#                 print(f"Model file not found: {model_filename}")
#                 return None
            
#             # Cargar el modelo
#             best_model = joblib.load(model_filename)
            
#             X_train = self.train[['week', 'year']]
#             y_train = self.train[self.product]
#             X_test = self.test[['week', 'year']]
#             y_test = self.test[self.product]

#             y_pred = y_test.copy()
#             last_date = y_pred.index[-1]
#             final_date = last_date + pd.DateOffset(months=self.months_to_predict)
#             new_dates = pd.date_range(start=last_date, end=final_date, freq='W-SUN')[1:]
#             df_new_dates = pd.DataFrame(index=new_dates)
#             df_new_dates['week'] = df_new_dates.index.isocalendar().week
#             df_new_dates['year'] = df_new_dates.index.year

#             predictions = best_model.predict(df_new_dates[['week', 'year']])
#             df_new_dates['predicted_quantity'] = predictions

#             df_pred_monthly = df_new_dates.resample('M').sum()

#             return df_pred_monthly

#         except Exception as e:
#             print(f"Error during prediction: {e}")
#             return None

#     def find_product_name(self, id_product):
#         URL = f"http://127.0.0.1:8010/TODOREPUESTOS/search"
#         payload = {"model": "product.product",
#                     "domain": [("id", "=", id_product)],
#                     "fields_names": ["name"],
#                     "context": {"company": 1, "user": 1}
#                 }
#         response = api.request_post_data(URL, payload)
#         product = response.json().pop()
#         return product["name"]
