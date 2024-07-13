import setup_db

query_product_str = """
    CREATE TABLE IF NOT EXISTS product (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT
    )
"""

query_last_update_stock_str = """
    CREATE TABLE IF NOT EXISTS last_update_stock_move (
        id SERIAL PRIMARY KEY,
        last_update_date DATE NOT NULL
    )
"""

query_stock_str = """
    CREATE TABLE IF NOT EXISTS stock_move (
        id SERIAL PRIMARY KEY,
        id_move INTEGER NOT NULL,
        product VARCHAR(255) NOT NULL,
        effective_date DATE NOT NULL,
        origin VARCHAR(50) NOT NULL,
        quantity INTEGER NOT NULL
    )
"""

model_tables = {
    'stock_ia': {
        # 'query_product': query_product_str,
        'query_stock': query_stock_str,
        'query_last_update_stock':query_last_update_stock_str
    },
}

def main(args):
    database_name = args.get('database_name')
    model_tables = args.get('model_ia')
    setup_db.create_database(database_name)
    setup_db.create_table(database_name, model_tables)

if __name__ == "__main__":
  args = {
        'database_name': 'TODOREPUESTOSIA2',
        'model_ia': model_tables['stock_ia']
    }
  main(args)