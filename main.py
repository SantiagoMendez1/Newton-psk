from projects.stock_ia.stock_move_etl import StockMoveEtl
from projects.stock_ia.stock_move_model import StockMoveModel
from projects.hotel_ia.hotel_etl import HotelEtl
from projects.hotel_ia.hotel_model import HotelModeL

DB = "TODOREP2"

def main():
    #stock_move_etl = StockMoveEtl(DB)
    #stock_move_etl.run()
    stock_move_model = StockMoveModel(DB)
    stock_move_model.pre_processing()
    # stock_move_model.model_hyperparam_tuning(run=True)
    df = stock_move_model.prediction(61578, 2)
    result = df.to_json(orient="index", indent=4)
    print(result)
    
    # -------------------- implemetation of hotel ------------------------
    
    # hotel_etl = HotelEtl(DB)
    # hotel_etl.run()
    #hotel_model = HotelModeL(DB)
    #x_train, x_test, y_train, y_test = hotel_model.split_train_data()

if __name__ == "__main__":
    main()
