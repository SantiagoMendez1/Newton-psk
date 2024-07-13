from core.stock_move.stock_move_etl import StockMoveEtl
from setup_db import create_database, create_table, connect_to_database

URL = 'http://127.0.0.1:8010/TODOREPUESTOS/search'

def main():
    stock_move_etl = StockMoveEtl(URL)
    stock_move_etl.run()

if __name__ == "__main__":
    main()
