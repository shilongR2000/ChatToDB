import random
from sqlalchemy import create_engine, text
import os


def get_sql_database_info_full(engine):
    """
    Fetch basic information for all tables in the database by running SELECT * FROM each table LIMIT 3.
    
    Args:
        engine: SQLAlchemy engine connected to the database.
        
    Returns:
        dict: A dictionary containing table names, their columns, and sample data.
    """
    try:
        database_info = {}
        with engine.connect() as conn:
            # Fetch all table names
            result = conn.execute(text("SHOW TABLES"))
            table_names = [row[0] for row in result]

            for table in table_names:
                # Get a sample of data from the table
                sample_query = f"SELECT * FROM {table} LIMIT 3"
                result = conn.execute(text(sample_query))
                rows = result.fetchall()
                columns = result.keys()

                # Determine column types based on sample data
                column_info = {}
                for idx, col in enumerate(columns):  # Use index instead of key access
                    sample_values = [row[idx] for row in rows if row[idx] is not None]
                    if all(isinstance(value, (int, float)) for value in sample_values):
                        column_info[col] = 'int'
                    else:
                        column_info[col] = 'str'

                database_info[table] = {
                    "columns": column_info,
                    "sample_data": [dict(zip(columns, row)) for row in rows]
                }

        return database_info
    except Exception as e:
        print(f"[ERROR] Error in get_sql_database_info: {str(e)}")
        return {}




def get_sql_example_with_where(database_info):
    """
    Generate a random SQL query example with a WHERE clause based on database information.
    
    Args:
        database_info: Dictionary containing table and column information with data types.
        
    Returns:
        str: A randomly generated SQL query example with a WHERE clause.
    """
    import random

    try:
        # Randomly select a keyword for the query
        action = random.choice(["find", "get", "show", "list", "display", "select"])

        # Flatten all columns across tables and filter by type
        all_columns = []
        for table, info in database_info.items():
            for col_name, col_type in info["columns"].items():
                all_columns.append((table, col_name, col_type))

        # Randomly choose between numeric and string-based conditions
        condition_type = random.choice(["number", "string"])

        if condition_type == "number":
            # Filter columns of numeric type
            numeric_columns = [
                (table, col) for table, col, col_type in all_columns if col_type == "int" and "id" not in col.lower()
            ]
            if not numeric_columns:
                raise ValueError("No numeric columns available for generating a WHERE clause.")

            table, column = random.choice(numeric_columns)
            condition_keyword = random.choice(["greater than", "more than", "less than", "under"])
            
            # Calculate a condition based on sample data
            sample_data = [row[column] for row in database_info[table]["sample_data"] if column in row]
            if not sample_data:
                condition_value = random.randint(1, 100)  # Fallback random number
            else:
                condition_value = round(sum(sample_data) / len(sample_data))  # Use average of sample data

            condition = f"{column} {condition_keyword} {condition_value}"
        
        else:  # String condition
            # Filter columns of string type
            string_columns = [
                (table, col) for table, col, col_type in all_columns if col_type == "str"
            ]
            if not string_columns:
                raise ValueError("No string columns available for generating a WHERE clause.")

            table, column = random.choice(string_columns)
            condition_keyword = random.choice(["is", "equals", "starts with", "ends with", "contains"])
            
            # Get a random sample value from the column
            sample_data = [row[column] for row in database_info[table]["sample_data"] if column in row and isinstance(row[column], str)]
            if not sample_data:
                condition_value = "example"  # Fallback random string
            else:
                condition_value = random.choice(sample_data)
            
            if condition_keyword == "contains":
                condition = f"{column} {condition_keyword} '{condition_value}'"
            elif condition_keyword == "starts with":
                condition = f"{column} {condition_keyword} '{condition_value}'"
            elif condition_keyword == "ends with":
                condition = f"{column} {condition_keyword} '{condition_value}'"
            else:
                condition = f"{column} {condition_keyword} '{condition_value}'"

        # Formulate the suggested query
        connector = random.choice(["that", "with"])
        suggested_message = f"{action} {table} {connector} {condition}"

        return suggested_message
    except Exception as e:
        print(f"[ERROR] Error in get_sql_example_with_where: {str(e)}")
        return "Could not generate an example query."



def get_sql_example_with_join(database_info):
    """
    Generate a random example query involving JOINs between tables based on joinable columns.

    Args:
        database_info (dict): Information about the database, including tables and their columns.

    Returns:
        str: A randomly generated example query involving JOINs.
    """
    # Randomly select an action keyword
    action = random.choice(["find", "get", "show", "list", "display", "select"])

    # Find joinable tables based on common columns
    joinable_tables = []
    for table1, info1 in database_info.items():
        for table2, info2 in database_info.items():
            if table1 != table2:
                # Find common columns between the two tables
                common_columns = set(map(str.lower, info1["columns"].keys())).intersection(
                    map(str.lower, info2["columns"].keys())
                )
                if common_columns:
                    joinable_tables.append((table1, table2, common_columns))

    # Ensure there are joinable tables
    if not joinable_tables:
        return "No joinable tables found in the database."

    # Randomly select a pair of joinable tables
    table1, table2, common_columns = random.choice(joinable_tables)
    common_column = random.choice(list(common_columns)).capitalize()  # Capitalize to match original case

    # Define message patterns
    pattern_a = f"{action} {table1} with {table2} details"
    pattern_b = f"{action} {table2} with their {table1} info"

    # Randomly select a message pattern
    suggested_message = random.choice([pattern_a, pattern_b])

    return suggested_message


def get_sql_example_with_groupby_having(database_info):
    """
    Generate a random example query involving GROUP BY and HAVING clauses based on recognition patterns.
    
    Args:
        database_info (dict): Information about the database, including tables and their columns.
    
    Returns:
        str: A randomly generated example query with GROUP BY and HAVING.
    """
    try:
        # Randomly select an action keyword
        action = random.choice(["find", "get", "show", "list", "display", "calculate"])
        
        # Aggregation functions that match self.agg_patterns
        agg_functions = ["count", "sum", "avg", "min", "max"]
        agg_function = random.choice(agg_functions)
        
        # Group by connectors that match self.group_by_patterns
        group_connectors = ["per", "in each", "for each"]
        group_connector = random.choice(group_connectors)
        
        # Flatten all columns across tables and filter by type
        all_columns = []
        for table, info in database_info.items():
            for col_name, col_type in info["columns"].items():
                all_columns.append((table, col_name, col_type))
        
        # Find suitable tables with both numeric and string columns
        suitable_tables = set()
        for table, col_name, col_type in all_columns:
            if col_type == "int" and "id" in col_name.lower():  # Potential aggregation column
                for t, c, ct in all_columns:
                    if t == table and ct == "str":  # Found a string column for grouping
                        suitable_tables.add(table)
                        break
        
        if not suitable_tables:
            raise ValueError("No suitable tables found for GROUP BY and HAVING.")
        
        # Select a random table
        table = random.choice(list(suitable_tables))
        
        # Get numeric columns (for aggregation) and string columns (for grouping)
        numeric_columns = [
            col for t, col, col_type in all_columns 
            if t == table and col_type == "int" and "id" in col.lower()
        ]
        string_columns = [
            col for t, col, col_type in all_columns 
            if t == table and col_type == "str"
        ]
        
        # Select random columns
        agg_column = random.choice(numeric_columns)
        group_column = random.choice(string_columns)
        
        # Generate comparison value
        comparison_value = random.randint(2, 5)
        
        # Comparison operators that match your patterns
        comparison_patterns = [
            "greater than",
            "more than",
            "above",
            "less than",
            "under",
            "below",
            "equal to",
            "exactly",
            "at least",
            "at most"
        ]
        
        comparison = random.choice(comparison_patterns)
        
        # Generate query using recognized patterns
        suggested_message = f"{action} {agg_function} {agg_column} {group_connector} {group_column} {comparison} {comparison_value}"
        
        return suggested_message
        
    except Exception as e:
        print(f"[ERROR] Error in get_sql_example_with_groupby_having: {str(e)}")
        return "Could not generate an example query."
    
    
def main():
    """
    Main function to test get_sql_database_info and get_sql_example_with_where.
    """
    # Replace with your database connection details
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '00112000')
    MYSQL_DB = 'mysql_551_enrollment'
    MYSQL_PORT = 3306
    engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")

    
    # Step 2: Retrieve database info
    print("Fetching database information...")
    database_info = get_sql_database_info_full(engine)
    if not database_info:
        print("No database information retrieved.")
        return

    # Display database information
    print("\nDatabase Information:")
    for table, info in database_info.items():
        print(f"Table: {table}")
        print("Columns:")
        for column, col_type in info["columns"].items():
            print(f"  {column}: {col_type}")
        print("Sample Data:")
        for row in info["sample_data"]:
            print(row)
        print()

    # Step 3: Generate an example query
    print("Generating an example query...")
    example_query = get_sql_example_with_where(database_info)
    print(f"\nGenerated where Query Example:\n{example_query}")

    example_query = get_sql_example_with_join(database_info)
    print(f"\nGenerated join Query Example:\n{example_query}")
    
    example_query = get_sql_example_with_groupby_having(database_info)
    print(f"\nGenerated join Query Example:\n{example_query}")


# Run the main function
if __name__ == "__main__":
    main()