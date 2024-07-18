from sqlalchemy import create_engine, text

def connect_to_database(database_name):
    try:
        engine = create_engine(f'postgresql://santiago:santiago@localhost:5432/{database_name}', isolation_level="AUTOCOMMIT")
        conn = engine.connect()
        print(f"Conexión exitosa a la base de datos '{database_name}'")
        return engine, conn
    except Exception as e:
        print(f"Error al conectar a la base de datos '{database_name}': {e}")
        return None, None

def create_database(database_name):
    try:
        engine, conn = connect_to_database("postgres")
        if conn is None:
            return
        conn.execute(text(f'CREATE DATABASE "{database_name}"'))
        print(f"Base de datos '{database_name}' creada correctamente")
    except Exception as e:
        print(f"Error al intentar crear la base de datos: {e}")
    finally:
        if conn is not None:
            conn.close()

def create_tables(database_name, model_ia_list, model_tables):
    try:
        create_database(database_name)
        engine, conn = connect_to_database(database_name)
        if conn is None:
            return

        for model_name in model_ia_list:
            model_ia = model_tables.get(model_name, {})
            for table_name, query in model_ia.items():
                conn.execute(text(query))
                print(f"Tabla '{table_name}' creada correctamente en la base de datos '{database_name}'")

    except Exception as e:
        print(f"Error al intentar crear las tablas: {e}")
    finally:
        if conn is not None:
            conn.close()
