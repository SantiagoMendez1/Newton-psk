import tools
import utils.db_utils as db_utils

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
        last_update_date TIMESTAMP NOT NULL
    )
"""

query_stock_str = """
    CREATE TABLE IF NOT EXISTS stock_move_ia (
        id SERIAL PRIMARY KEY,
        id_move INTEGER NOT NULL UNIQUE,
        product VARCHAR(255) NOT NULL,
        effective_date TIMESTAMP NOT NULL,
        origin VARCHAR(50) NOT NULL,
        quantity INTEGER NOT NULL
    )
"""

model_tables = {
    'stock_move_ia': {
        'query_product': query_product_str,
        'query_stock': query_stock_str,
        'query_last_update_stock': query_last_update_stock_str
    },
    'other_ia': {
    }
}


def main():
    config = tools.load_config()
    databases = config.get('databases', [])
    
    for db in databases:
        database_name = db['name']
        model_ia_list = db.get('models', [])
        print(f"Creating tables for database: {database_name} with models: {model_ia_list}")
        db_utils.create_tables(database_name, model_ia_list, model_tables)

if __name__ == "__main__":
    main()
