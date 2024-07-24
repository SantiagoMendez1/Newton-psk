from .stock_move_etl import StockMoveEtl

URL = 'http://127.0.0.1:8010/TODOREPUESTOS/search'

def run():
    stock_move_etl = StockMoveEtl(URL)
    stock_move_etl.run()

if __name__ == "__main__":
    run()
