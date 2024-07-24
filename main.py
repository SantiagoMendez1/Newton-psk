from projects.min_max_sm.stock_move_etl import StockMoveEtl
from projects.min_max_sm.model.stock_move_model import StockMoveModel

DB = "TODOREPUESTOS"

def main():
    # stock_move_etl = StockMoveEtl(DB)
    # stock_move_etl.run()
    stock_move_model = StockMoveModel(DB)
    stock_move_model.pre_processing()

if __name__ == "__main__":
    main()
