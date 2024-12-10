# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:33:07 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:59:55 2024

@author: Administrator
"""

from flask import Flask, render_template, request, jsonify, session
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from functools import wraps
import pandas as pd
import pymysql
import json
import os
import logging
from datetime import datetime
from bson import json_util

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MYSQL_DEFAULT_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '00112000',
    'port': 3306
}
MONGODB_URI = "mongodb+srv://shilongr:001120rsl@cluster0.pzrgm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from core_recognize_sql_v6 import get_sql_database_info, generate_sql_query#analyze_and_execute_sql_query, generate_sql_query, execute_sql_query  # Your MySQL query function
from core_recognize_mongo_v3_1 import extract_mongodb_schema, generate_mongodb_query#analyze_and_execute_mongo_query, generate_mongo_query, execute_mongo_query  # Your MongoDB query function
from core_generate_sample_sql_v1 import get_sql_example_with_where, get_sql_example_with_join, get_sql_example_with_groupby_having, get_sql_database_info_full
from core_generate_sample_mongo_v1 import get_mongo_example_with_match, get_mongo_example_with_lookup, get_mongo_example_with_group_by, extract_mongodb_schema_full

def connect_mysql(db_name=None):
   """Create MySQL connection"""
   try:
       # Base connection string
       connection_string = (
           f"mysql+pymysql://{MYSQL_DEFAULT_CONFIG['user']}:{MYSQL_DEFAULT_CONFIG['password']}@"
           f"{MYSQL_DEFAULT_CONFIG['host']}:{MYSQL_DEFAULT_CONFIG['port']}"
       )
       
       # Add database name if provided
       if db_name:
           connection_string += f"/{db_name}"

       engine = create_engine(connection_string)
       return engine
       
   except Exception as e:
       logger.error(f"MySQL connection error: {str(e)}")
       return None
    
    
def connect_mongodb():
    """Create MongoDB connection"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection
        return client
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index_04.html')

@app.route('/switch_db', methods=['POST'])
def switch_db():
    """Switch between MySQL and MongoDB"""
    try:
        db_type = request.form.get('db_type')
        if db_type not in ['mysql', 'mongodb']:
            return jsonify(error="Invalid database type"), 400

        session['db_type'] = db_type
        if db_type == 'mysql':
            engine = connect_mysql()
            if not engine:
                return jsonify(error="Could not connect to MySQL"), 500
        else:
            client = connect_mongodb()
            if not client:
                return jsonify(error="Could not connect to MongoDB"), 500

        return jsonify(success=True)
    except Exception as e:
        logger.error(f"Error in switch_db: {str(e)}")
        return jsonify(error=str(e)), 500

@app.route('/list_databases', methods=['GET'])
def list_databases():
    """List available databases"""
    try:
        if session.get('db_type') == 'mysql':
            engine = connect_mysql()
            if not engine:
                return jsonify(error="MySQL connection failed"), 500
            with engine.connect() as conn:
                result = conn.execute(text("SHOW DATABASES"))
                databases = [row[0] for row in result]
        else:
            client = connect_mongodb()
            if not client:
                return jsonify(error="MongoDB connection failed"), 500
            databases = client.list_database_names()
        
        return jsonify(databases=databases)
    except Exception as e:
        logger.error(f"Error in list_databases: {str(e)}")
        return jsonify(error=str(e)), 500

@app.route('/database_info/<db_name>')
def database_info(db_name):
    """Get database schema information"""
    try:
        if session.get('db_type') == 'mysql':
            engine = connect_mysql()
            if not engine:
                return jsonify(error="MySQL connection failed"), 500
            
            tables = []
            with engine.connect() as conn:
                conn.execute(text(f"USE {db_name}"))
                result = conn.execute(text("SHOW TABLES"))
                table_names = [row[0] for row in result]
                
                for table_name in table_names:
                    result = conn.execute(text(f"DESCRIBE {table_name}"))
                    columns = [{"name": row[0], "type": row[1]} for row in result]
                    tables.append({
                        "name": table_name,
                        "columns": columns
                    })
        else:
            client = connect_mongodb()
            if not client:
                return jsonify(error="MongoDB connection failed"), 500
            
            db = client[db_name]
            tables = []
            
            for collection_name in db.list_collection_names():
                sample = db[collection_name].find_one()
                columns = []
                if sample:
                    for key, value in sample.items():
                        if key != '_id':  # Skip MongoDB ID field
                            columns.append({
                                "name": key,
                                "type": type(value).__name__
                            })
                
                tables.append({
                    "name": collection_name,
                    "columns": columns
                })

        return jsonify({
            "database": db_name,
            "tables": tables
        })

    except Exception as e:
        logger.error(f"Error in database_info: {str(e)}")
        return jsonify(error=str(e)), 500

@app.route('/query', methods=['POST'])
def query():
    """Execute database query"""
    try:
        user_message = request.form.get('user_message')
        db_name = request.form.get('db_name')
        
        if user_message == 'hello':
            response_data = {'message': "Hello, nice to meet you!"}
            return jsonify(response_data)
    
        elif user_message in [
            'example with where', 'example with join', 
            'example with groupby having', 'example with match', 
            'example with lookup', 'example with group by', 'get some example', 'help']:
            db_type = session.get('db_type')
    

            if db_type == 'mysql':
                engine = connect_mysql(db_name)
                database_info_full = get_sql_database_info_full(engine)
                database_info = get_sql_database_info(engine)
            
                # Dispatch functions for MySQL
                sql_query_dispatch = {
                    'example with where': lambda: {
                        'result': (result := get_sql_example_with_where(database_info_full)),
                        'query': generate_sql_query(result, database_info)
                    },
                    'example with join': lambda: {
                        'result': (result := get_sql_example_with_join(database_info_full)),
                        'query': generate_sql_query(result, database_info)
                    },
                    'example with groupby having': lambda: {
                        'result': (result := get_sql_example_with_groupby_having(database_info_full)),
                        'query': generate_sql_query(result, database_info)
                    },
                }
            
                if user_message == 'get some example' or user_message == 'help':
                    # Generate and execute examples
                    example_data = []
                    for query_name, func in sql_query_dispatch.items():
                        result_data = func()  # Call once to get both the query and result
                        example_data.append({"Name": "message", "Content": result_data['result']})
                        example_data.append({"Name": "query", "Content": result_data['query']})
            
                    # Create a DataFrame for the response
                    result_table = pd.DataFrame(example_data)
            
                    response_data = {
                        'message': f"Found {len(result_table)} rows.",
                        'table': result_table.to_html(
                            classes='table table-bordered table-striped',
                            index=False
                        )
                    }
            
                    return jsonify(response_data)
            
                elif user_message in sql_query_dispatch:
                    # Generate and execute the specific example
                    result_data = sql_query_dispatch[user_message]()  # Call once to get both
                    example_data = [
                        {"Name": "message", "Content": result_data['result']},
                        {"Name": "query", "Content": result_data['query']}
                    ]
            
                    response_data = {
                        'message': f"Query and result for {user_message}",
                        'table': pd.DataFrame(example_data).to_html(
                            classes='table table-bordered table-striped',
                            index=False
                        )
                    }
                    return jsonify(response_data)


    
            elif db_type == 'mongodb':
                client = connect_mongodb()
                if not client:
                    return jsonify(error="MongoDB connection failed"), 500
            
                db = client[db_name]
                schema_info_full = extract_mongodb_schema_full(db)
                schema_info = extract_mongodb_schema(db)
            
                # Dispatch functions for MongoDB
                mongo_query_dispatch = {
                    'example with match': lambda: {
                        'result': (result := get_mongo_example_with_match(schema_info_full)),
                        'query': (lambda: (
                            collection := (query_info := generate_mongodb_query(result, schema_info))[0],
                            pipeline := query_info[1],
                            f"db['{collection}'].aggregate({json.dumps(pipeline)})"
                        )[-1])()
                    },
                    'example with lookup': lambda: {
                        'result': (result := get_mongo_example_with_lookup(schema_info_full)),
                        'query': (lambda: (
                            collection := (query_info := generate_mongodb_query(result, schema_info))[0],
                            pipeline := query_info[1],
                            f"db['{collection}'].aggregate({json.dumps(pipeline)})"
                        )[-1])()
                    },
                    'example with group by': lambda: {
                        'result': (result := get_mongo_example_with_group_by(schema_info_full)),
                        'query': (lambda: (
                            collection := (query_info := generate_mongodb_query(result, schema_info))[0],
                            pipeline := query_info[1],
                            f"db['{collection}'].aggregate({json.dumps(pipeline)})"
                        )[-1])()
                    }
                }
            
                if user_message == 'get some example' or user_message == 'help':
                    # Generate and execute examples
                    example_data = []
                    for query_name, func in mongo_query_dispatch.items():
                        result_data = func()  # Call once to get both query and result
                        example_data.append({"Name": "message", "Content": json.dumps(result_data['result'], indent=2)})
                        example_data.append({"Name": "query", "Content": result_data['query']})
            
                    # Create a DataFrame for the response
                    result_table = pd.DataFrame(example_data)
            
                    response_data = {
                        'message': f"Found {len(result_table)} rows.",
                        'table': result_table.to_html(
                            classes='table table-bordered table-striped',
                            index=False
                        )
                    }
            
                    return jsonify(response_data)
            
                elif user_message in mongo_query_dispatch:
                    # Generate and execute the specific example
                    result_data = mongo_query_dispatch[user_message]()  # Call once to get both
                    example_data = [
                        {"Name": "message", "Content": json.dumps(result_data['result'], indent=2)},
                        {"Name": "query", "Content": result_data['query']}
                    ]
            
                    response_data = {
                        'message': f"Query and result for {user_message}",
                        'table': pd.DataFrame(example_data).to_html(
                            classes='table table-bordered table-striped',
                            index=False
                        )
                    }
                    return jsonify(response_data)
                
        else:
            
            if not user_message or not db_name:
                return jsonify(error="Missing query or database name"), 400
    
            if session.get('db_type') == 'mysql':
                engine = connect_mysql(db_name)  # Pass db_name directly to connect_mysql
                if not engine:
                    return jsonify(error="MySQL connection failed"), 500
                
                database_info = get_sql_database_info(engine)
                sql_query = generate_sql_query(user_message, database_info)
                result_table = pd.read_sql(sql_query, engine)
                success = 1
                #sql_query, success, result_table = analyze_and_execute_sql_query(user_message, engine)
                
                response_data = {
                    'generated_query': f"Generated SQL: {sql_query}"
                }
                
                if not success:
                    response_data.update({
                        'error': "Query execution failed",
                        'message': "Query failed but generated SQL is shown above."
                    })
                    return jsonify(response_data), 500
                
                response_data.update({
                    'message': f"Found {len(result_table)} rows.",
                    'table': result_table.to_html(
                        classes='table table-bordered table-striped',
                        index=False
                    )
                })
                return jsonify(response_data)
    
            else:
                client = connect_mongodb()
                if not client:
                    return jsonify(error="MongoDB connection failed"), 500
                
                db = client[db_name]
                
                schema_info = extract_mongodb_schema(db)
                collection, pipeline = generate_mongodb_query(user_message, schema_info)
                
                # Execute query
                result = list(db[collection].aggregate(pipeline))
                result_df = pd.json_normalize(json.loads(json_util.dumps(result)))
                
                # Format the query string
                pipeline_str = json.dumps(pipeline, indent=2)
                mongo_query = f"db.{collection}.aggregate({pipeline_str})"
                
                success = 1
                
                # mongo_query = collection + '.' + pipeline
                # mongo_query, success, result_df = analyze_and_execute_mongo_query(user_message, db)
                
                response_data = {
                    'generated_query': f"Generated MongoDB Query: {mongo_query}"
                }
                
                if not success:
                    response_data.update({
                        'error': "Query execution failed",
                        'message': "Query failed but generated MongoDB query is shown above."
                    })
                    return jsonify(response_data), 500
                
                response_data.update({
                    'message': f"Found {len(result_df)} documents.",
                    'table': result_df.to_html(
                        classes='table table-bordered table-striped',
                        index=False
                    )
                })
                return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'generated_query': f"Generated query: {sql_query if 'sql_query' in locals() else mongo_query if 'mongo_query' in locals() else 'Query generation failed'}"
        }), 500


# Add this route to your Flask application
@app.route('/create_database', methods=['POST'])
def create_database():
    """Create a new database in MySQL or MongoDB"""
    try:
        db_name = request.form.get('db_name')
        if not db_name:
            return jsonify(error="No database name provided"), 400
            
        # Validate database name (basic validation)
        if not db_name.replace('_', '').isalnum():
            return jsonify(error="Database name can only contain letters, numbers, and underscores"), 400
            
        if session.get('db_type') == 'mysql':
            engine = connect_mysql()
            if not engine:
                return jsonify(error="MySQL connection failed"), 500
                
            try:
                with engine.connect() as connection:
                    # Create new database
                    connection.execute(text(f"CREATE DATABASE `{db_name}`"))
                    logger.info(f"Created MySQL database: {db_name}")
                return jsonify(success=True, message=f"MySQL database '{db_name}' created successfully!")
                
            except Exception as e:
                logger.error(f"MySQL database creation error: {str(e)}")
                return jsonify(error=f"Error creating MySQL database: {str(e)}"), 500
                
        else:  # MongoDB
            client = connect_mongodb()
            if not client:
                return jsonify(error="MongoDB connection failed"), 500
                
            try:
                # In MongoDB, databases are created automatically when you first store data
                # We'll create a temporary collection to ensure the database exists
                db = client[db_name]
                db.create_collection('_temp')
                db.drop_collection('_temp')
                logger.info(f"Created MongoDB database: {db_name}")
                return jsonify(success=True, message=f"MongoDB database '{db_name}' created successfully!")
                
            except Exception as e:
                logger.error(f"MongoDB database creation error: {str(e)}")
                return jsonify(error=f"Error creating MongoDB database: {str(e)}"), 500
                
    except Exception as e:
        logger.error(f"Database creation error: {str(e)}")
        return jsonify(error=str(e)), 500
    
    
@app.route('/upload_data', methods=['POST'])
def upload_file():
    # Get database name from form data
    db_name = request.form.get('db_name')
    if not db_name:
        return jsonify({'error': 'No database name provided'}), 400

    logger.info(f"Uploading to database: {db_name}")
    logger.info(f"Database type: {session.get('db_type')}")

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the file content first
        file_content = file.read()
        
        # Check if file is empty
        if len(file_content) == 0:
            return jsonify({'error': 'File is empty'}), 400
            
        # Reset file pointer to beginning
        file.seek(0)
        
        # Handle MySQL database uploads (xlsx and csv only)
        if session.get('db_type') == 'mysql':
            # Verify MySQL connection is active
            engine = connect_mysql()
            if not engine:
                return jsonify({'error': 'MySQL connection failed'}), 500

            try:
                if file.filename.endswith('.xlsx'):
                    df = pd.read_excel(file)
                    logger.info("Successfully read Excel file")
                elif file.filename.endswith('.csv'):
                    # Try different encodings and delimiters
                    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1']
                    delimiters = [',', ';', '\t', '|']
                    df = None
                    last_error = None

                    for encoding in encodings:
                        if df is not None:
                            break
                        for delimiter in delimiters:
                            try:
                                file.seek(0)
                                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding)
                                if not df.empty:
                                    logger.info(f"Successfully read CSV with encoding {encoding} and delimiter {delimiter}")
                                    break
                            except Exception as e:
                                last_error = str(e)
                                continue

                    if df is None:
                        return jsonify({'error': f'Could not read CSV file. Last error: {last_error}'}), 400
                else:
                    return jsonify({'error': 'For MySQL databases, only .xlsx and .csv files are supported'}), 400

                # Verify data was read successfully
                if df is None or df.empty:
                    return jsonify({'error': 'No data found in file'}), 400

                # Get table name from filename (remove extension)
                table_name = os.path.splitext(file.filename)[0]
                logger.info(f"Creating table: {table_name}")

                # Clean column names
                df.columns = df.columns.str.replace('[^0-9a-zA-Z_]', '_', regex=True)
                
                # Store data information
                columns = df.columns.tolist()
                unique_items = {col: int(df[col].nunique()) for col in columns}
                
                # Upload to MySQL
                try:
                    with engine.connect() as connection:
                        # Create database if it doesn't exist
                        connection.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
                        
                        # Use the database
                        connection.execute(text(f"USE `{db_name}`"))
                        logger.info(f"Using database: {db_name}")
                        
                        # Drop existing table
                        connection.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
                        
                        # Commit the transaction
                        connection.commit()
                    
                    # Create table and insert data
                    df.to_sql(
                        name=table_name,
                        con=engine,
                        schema=db_name,
                        if_exists='replace',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                    
                    upload_message = f"Data uploaded to MySQL table '{table_name}' successfully!"
                    logger.info(upload_message)
                    
                except Exception as e:
                    logger.error(f"MySQL upload error: {str(e)}")
                    return jsonify({'error': f'Error uploading to MySQL: {str(e)}'}), 400

                # Get table preview
                preview_df = df.head()
                table_html = preview_df.to_html(classes='table', escape=False)
                unique_items_str = '<br>'.join([f"{col}: {count}" for col, count in unique_items.items()])

                # Get current tables
                tables = []
                with engine.connect() as connection:
                    connection.execute(text(f"USE `{db_name}`"))
                    result = connection.execute(text("SHOW TABLES"))
                    tables = [row[0] for row in result]

                return jsonify({
                    'message': f"{upload_message}<br>Current Tables: {', '.join(tables)}",
                    'table': f"<strong>Preview of {file.filename}:</strong><br>{table_html}",
                    'unique_items': f"<strong>Unique Items per Column:</strong><br>{unique_items_str}"
                })

            except Exception as e:
                logger.error(f"File processing error: {str(e)}")
                return jsonify({'error': f'Error processing file: {str(e)}'}), 400


        # Handle MongoDB database uploads (json only)
        elif session.get('db_type') == 'mongodb':
            if not file.filename.endswith('.json'):
                return jsonify({'error': 'For MongoDB databases, only .json files are supported'}), 400
                
            try:
                client = connect_mongodb()
                db = client[db_name]
                
                content = file_content.decode('utf-8')
                data = json_util.loads(content)
                
                # Handle both single document and array of documents
                if not isinstance(data, list):
                    data = [data]
                    
                # Convert to DataFrame for preview
                df = pd.json_normalize(data)
                
                # Store in MongoDB
                collection_name = os.path.splitext(file.filename)[0]
                collection = db[collection_name]
                
                # Drop existing collection if it exists
                collection.drop()
                
                # Insert documents
                collection.insert_many(data)
                upload_message = f"Data uploaded to MongoDB collection '{collection_name}' successfully!"
                
                # Prepare MongoDB preview
                preview_df = df.head()
                table_html = preview_df.to_html(classes='table', escape=False)
                unique_items = {col: df[col].nunique() for col in df.columns}
                unique_items_str = '<br>'.join([f"{col}: {count}" for col, count in unique_items.items()])
                
                # Get current collections
                collections = db.list_collection_names()
                
                return jsonify({
                    'message': f"{upload_message}<br>Current Collections: {', '.join(collections)}",
                    'table': f"<strong>Preview of {file.filename}:</strong><br>{table_html}",
                    'unique_items': f"<strong>Unique Items per Column:</strong><br>{unique_items_str}"
                })
                
            except Exception as e:
                return jsonify({'error': f'Error processing JSON file: {str(e)}'}), 400

        else:
            return jsonify({'error': 'Invalid database type'}), 400

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f"Error processing file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)