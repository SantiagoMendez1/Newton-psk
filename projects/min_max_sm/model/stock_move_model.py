import utils.db_utils as db
import utils.api_utils as api
import pandas as pd
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV

class StockMoveModel:
    def __init__(self, database):
        self.database = database
        self.engine, self.conn = db.connect_to_database(self.database)
        self.data = self.load_data()

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
            pivot_df = filtered_df.pivot_table(index='effective_date', columns='product', values='quantity', aggfunc='sum', fill_value=0)
            pivot_df = pivot_df.reindex(columns=self.filter_products)
            compl_date_day = pd.date_range(start=pivot_df.index.min(), end=pivot_df.index.max(), freq='D')
            df_compl_day = pd.DataFrame({'effective_date': compl_date_day})
            one_prod_df_all_date = pd.merge(df_compl_day, pivot_df, on='effective_date', how='left').fillna(0)
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
            print(f"KeyError: {e}. Ensure 'product' and 'quantity' are columns in the data.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def model_hyperparam_tuning(self, run=False):
        """
        Perform hyperparameter tuning for a RandomForestRegressor model.
        
        Args:
        run (bool): Flag to indicate whether to perform hyperparameter tuning or not. If False, attempts to load from file.
        """
        products_to_train = self.filter_products[0:1]
        if run is False:
            try:
                self.query = 'SELECT id_product, hyperparams from product_trained'
                self.result_query_best_params = db.execute_query(self.database, self.query)
                print(self.result_query_best_params.all())
            except:
                run = True
        print(products_to_train)
        if run:
            self.all_best_params = []
            for t in products_to_train:
                X_train = self.train[['week', 'year']]
                y_train = self.train[t]

                X_test = self.test[['week', 'year']]
                y_test = self.test[t]

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
                json.dumps(items_params)
                print(type(items_params))
                product_trained = {
                    "id_product": t,
                    "name":"example name",
                    "hyperparams":items_params
                }
                self.all_best_params.append(product_trained)
            df_product_trained = pd.DataFrame(self.all_best_params)
            print(df_product_trained)
            df_product_trained.to_sql(name="product_trained", con=self.engine,                  if_exists='append', index=False)
                # self.all_best_params["id_product"] = t
                # self.all_best_params["hyperparams"] = items_params
            # print(self.all_best_params)
            
            return None

    def find_product(self, id_product):
        URL = f"http://127.0.0.1:8010/{self.database}/search"
        payload = [{
                    "model": "product_product",
                    "domain": [("id", "=", id_product)],
                    "fields_names": ["product.name"],
                    "context": {"company": 1, "user": 1}
                }]
        return api.request_post_data()