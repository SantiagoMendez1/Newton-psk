import tools
import utils.db_utils as db

query_product_str = """
    CREATE TABLE IF NOT EXISTS products_trained (
        id SERIAL PRIMARY KEY,
        id_product INTEGER NOT NULL UNIQUE,
        name VARCHAR(255) NOT NULL,
        hyperparams JSONB
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
    
    for database in databases:
        database_name = database['name']
        model_ia_list = database.get('models', [])
        db.create_database(database_name)
        db.create_tables(database_name, 
                         model_ia_list, 
                         model_tables)

if __name__ == "__main__":
    main()
