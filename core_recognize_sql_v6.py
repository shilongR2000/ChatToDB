# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:10:24 2024

@author: Administrator
"""

import re
from typing import Dict, List, Optional
from fuzzywuzzy import process
from sqlalchemy import create_engine, inspect, MetaData
from typing import Dict, List
import re

class QueryRecognizer:
    def __init__(self):
        # Initialize patterns from your existing code
        self.agg_patterns = {
            'average': 'AVG',
            'avg': 'AVG',
            'min': 'MIN',
            'max': 'MAX',
            'sum': 'SUM',
            'total': 'SUM',
            'count': 'COUNT'
        }
        
        self.where_patterns = {
            r'([\w_]+)\s+is\s+("([^"]+)"|\b\w+\b)': lambda f, v: f"{f} = '{v.strip()}'",  # Matches one word or quoted string
            r'([\w_]+)\s+equals?\s+("([^"]+)"|\b\w+\b)': lambda f, v: f"{f} = '{v.strip()}'",  # Matches one word or quoted string
            r'([\w_]+)\s+greater\s+than\s+(\d+(?:\.\d+)?)': lambda f, v: f"{f} > {v}",  # Matches numeric values
            r'([\w_]+)\s+more\s+than\s+(\d+(?:\.\d+)?)': lambda f, v: f"{f} > {v}",  # Matches numeric values
            r'([\w_]+)\s+less\s+than\s+(\d+(?:\.\d+)?)': lambda f, v: f"{f} < {v}",  # Matches numeric values
            r'([\w_]+)\s+under\s+(\d+(?:\.\d+)?)': lambda f, v: f"{f} < {v}",  # Matches numeric values
            r'([\w_]+)\s+contains?\s+("([^"]+)"|\b\w+\b)': lambda f, v: f"{f} LIKE '%%{v.strip()}%%'",  # Matches one word or quoted string
            r'([\w_]+)\s+starts?\s+with\s+("([^"]+)"|\b\w+\b)': lambda f, v: f"{f} LIKE '{v.strip()}%%'",  # Matches one word or quoted string
            r'([\w_]+)\s+ends?\s+with\s+("([^"]+)"|\b\w+\b)': lambda f, v: f"{f} LIKE '%%{v.strip()}'"  # Matches one word or quoted string
        }

        
        self.group_by_patterns = [
            r'\bper\b\s+([\w_]+)',
            r'\bin each\b\s+([\w_]+)',
            r'\bfor each\b\s+([\w_]+)'
        ]
        
        self.join_patterns = [
            r'with\s+([\w_]+)\s+details?',
            r'join\s+with\s+([\w_]+)',
            r'include\s+([\w_]+)',
            r'related\s+([\w_]+)',
            r'its\s+([\w_]+)',
            r'their\s+([\w_]+)'
        ]
        
        self.orderby_patterns = [
            (r'order\s+by\s+([\w_]+)\s+desc(?:ending)?', 'DESC'),
            (r'order\s+by\s+([\w_]+)\s+asc(?:ending)?', 'ASC'),
            (r'sort\s+by\s+([\w_]+)\s+desc(?:ending)?', 'DESC'),
            (r'sort\s+by\s+([\w_]+)\s+asc(?:ending)?', 'ASC'),
            (r'order\s+by\s+([\w_]+)', 'ASC'),
            (r'sort\s+by\s+([\w_]+)', 'ASC')
        ]

        self.query_parts = {
            'select': [],
            'from': [],
            'join': [],
            'where': [],
            'group_by': [],
            'having': [],
            'order_by': []
        }

    def split_message_to_parts(self, message: str) -> List[str]:
        """Split message into parts while preserving quoted strings."""
        # Split by spaces but preserve quoted strings
        parts = []
        current_part = []
        in_quotes = False
        quote_char = None
        
        for char in message:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_part.append(char)
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_part.append(char)
            elif char.isspace() and not in_quotes:
                if current_part:
                    parts.append(''.join(current_part))
                    current_part = []
            else:
                current_part.append(char)
                
        if current_part:
            parts.append(''.join(current_part))
            
        return parts

    def recognize_patterns(self, message: str, database_info: Dict) -> Dict:
        """Recognize patterns in the message and map them to SQL query parts."""
        message_parts = self.split_message_to_parts(message.lower())
        
        # First, identify the main table and columns
        self._identify_table_and_columns(message_parts, database_info)
        
        # Recognize aggregations
        self._recognize_aggregations(message, database_info)
        
        # Recognize joins
        self._recognize_joins(message, database_info)
        
        # Recognize where conditions
        self._recognize_where_conditions(message, database_info)
        
        # Recognize group by
        self._recognize_group_by(message, database_info)
        
        # Recognize order by
        self._recognize_order_by(message, database_info)
        
        return self.query_parts

    def _identify_table_and_columns(self, message_parts: List[str], database_info: Dict):
        """Identify tables and columns from the message."""
        # Create a copy of message parts to modify
        remaining_parts = message_parts.copy()
        parts_to_remove = []
        
        # First try to find table names
        for i, part in enumerate(remaining_parts):
            # Clean part from quotes if present
            clean_part = part.strip('"\'')
            
            # Try to match with table names
            table_matches = process.extract(clean_part, list(database_info.keys()), limit=1)
            if table_matches and table_matches[0][1] >= 80:  # 80% similarity threshold
                matched_table = table_matches[0][0]
                if matched_table not in self.query_parts['from']:
                    self.query_parts['from'].append(matched_table)
                    parts_to_remove.append(part)
    
        # Remove matched table parts
        remaining_parts = [part for part in remaining_parts if part not in parts_to_remove]
                    
        # Then try to find columns for the identified tables
        if clean_part in database_info:
            if clean_part not in self.query_parts['from']:
                self.query_parts['from'].append(clean_part)
                # Add '*' as a fallback for columns
                if not self.query_parts['select']:
                    self.query_parts['select'].append('*')
                        


    def _fuzzy_match_column(self, column_name: str, database_info: Dict, similarity_threshold: int = 80) -> Optional[str]:
        """
        Fuzzy match a column name against all available columns in the database.
        Returns the best matching column name or None if no good match is found.
        """
        if not self.query_parts['from']:
            return None
            
        main_table = self.query_parts['from'][0]
        column_matches = process.extract(
            column_name,
            database_info[main_table]['columns'],
            limit=1
        )
        
        if column_matches and column_matches[0][1] >= similarity_threshold:
            return column_matches[0][0]
        return None
    
    def _fuzzy_match_table(self, table_name: str, database_info: Dict, similarity_threshold: int = 80) -> Optional[str]:
        """
        Fuzzy match a table name against all tables in the database.
        Returns the best matching table name or None if no good match is found.
        """
        table_matches = process.extract(
            table_name,
            list(database_info.keys()),
            limit=1
        )
        
        if table_matches and table_matches[0][1] >= similarity_threshold:
            return table_matches[0][0]
        return None
    
    
    def _recognize_aggregations(self, message: str, database_info: Dict):
        """Recognize aggregation functions and comparison conditions in the message with fuzzy column matching."""
        words = message.lower().split()
        message = message.lower()
        
        have_groupby = 0
        # First handle regular aggregations
        for agg_word, agg_func in self.agg_patterns.items():
            if agg_word in words:
                idx = words.index(agg_word)
                if idx + 1 < len(words):
                    potential_column = words[idx + 1]
                    have_groupby = 1
                    # Special handling for COUNT
                    if agg_func == 'COUNT':
                        self.query_parts['select'] = [f"COUNT(1) as count"]
                    else:
                        # For other aggregations, use the matched column
                        matched_column = self._fuzzy_match_column(potential_column, database_info)
                        if matched_column:
                            self.query_parts['select'].append(f"{agg_func}({matched_column})")
                    
                    words.pop(idx + 1)  # Remove the matched column
                    words.pop(idx)      # Remove the aggregation word
        
        # Then check for comparison patterns after GROUP BY
        if have_groupby == 1:
            # Define comparison patterns
            comparison_patterns = [
                # Greater than patterns
                (r'greater\s+th[ae]n\s+(\d+)', '>'),
                (r'more\s+th[ae]n\s+(\d+)', '>'),
                (r'above\s+(\d+)', '>'),
                (r'exceeds?\s+(\d+)', '>'),
                (r'over\s+(\d+)', '>'),
                (r'>\s*(\d+)', '>'),
                
                # Less than patterns
                (r'less\s+th[ae]n\s+(\d+)', '<'),
                (r'under\s+(\d+)', '<'),
                (r'below\s+(\d+)', '<'),
                (r'fewer\s+th[ae]n\s+(\d+)', '<'),
                (r'<\s*(\d+)', '<'),
                
                # Equal patterns
                (r'equal\s+to\s+(\d+)', '='),
                (r'equals?\s+(\d+)', '='),
                (r'exactly\s+(\d+)', '='),
                (r'is\s+(\d+)', '='),
                (r'=\s*(\d+)', '='),
                
                # Greater than or equal patterns
                (r'at\s+least\s+(\d+)', '>='),
                (r'minimum\s+of\s+(\d+)', '>='),
                (r'not\s+less\s+th[ae]n\s+(\d+)', '>='),
                (r'>=\s*(\d+)', '>='),
                
                # Less than or equal patterns
                (r'at\s+most\s+(\d+)', '<='),
                (r'maximum\s+of\s+(\d+)', '<='),
                (r'not\s+more\s+th[ae]n\s+(\d+)', '<='),
                (r'<=\s*(\d+)', '<='),
                
                # Not equal patterns
                (r'not\s+equal\s+to\s+(\d+)', '!='),
                (r'different\s+from\s+(\d+)', '!='),
                (r'!=\s*(\d+)', '!='),
                (r'<>\s*(\d+)', '!=')
            ]
            
            # Check each pattern
            for pattern, operator in comparison_patterns:
                match = re.search(pattern, message)
                if match:
                    value = match.group(1)
                    if any(agg in select for select in self.query_parts.get('select', []) 
                          for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']):
                        # Important change: Use bare COUNT(1) in HAVING clause, not the aliased version
                        if 'COUNT(1) as count' in self.query_parts['select']:
                            self.query_parts.setdefault('having', []).append(f"COUNT(1) {operator} {value}")
                        else:
                            # For other aggregations, get the expression without the alias
                            agg_expr = next(select.split(' as ')[0] for select in self.query_parts['select'] 
                                      if any(agg in select for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']))
                            self.query_parts.setdefault('having', []).append(f"{agg_expr} {operator} {value}")
                    break
                    
                    
    def _recognize_joins(self, message: str, database_info: Dict):
        """Recognize join conditions in the message with fuzzy table matching."""
        for pattern in self.join_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                potential_table = match.group(1)
                matched_table = self._fuzzy_match_table(potential_table, database_info)
                if matched_table and matched_table not in self.query_parts['join']:
                    self.query_parts['join'].append(matched_table)
    
    def _recognize_where_conditions(self, message: str, database_info: Dict):
        """Recognize where conditions in the message with fuzzy column matching."""
        for pattern, formatter in self.where_patterns.items():
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                potential_field = match.group(1)
                value = match.group(2)
                
                matched_field = self._fuzzy_match_column(potential_field, database_info)
                if matched_field:
                    try:
                        # Create the condition with proper escaping
                        condition = formatter(matched_field, value)
                        if condition not in self.query_parts['where']:
                            self.query_parts['where'].append(condition)
                    except Exception as e:
                        print(f"Error formatting condition: {str(e)}")
                        continue
    
    def _recognize_group_by(self, message: str, database_info: Dict):
        """Recognize group by clauses in the message with fuzzy column matching."""
        for pattern in self.group_by_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                potential_field = match.group(1)
                matched_field = self._fuzzy_match_column(potential_field, database_info)
                if matched_field and matched_field not in self.query_parts['group_by']:
                    self.query_parts['group_by'].append(matched_field)
    
    def _recognize_order_by(self, message: str, database_info: Dict):
        """Recognize order by clauses in the message with fuzzy column matching."""
        for pattern, direction in self.orderby_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                potential_field = match.group(1)
                matched_field = self._fuzzy_match_column(potential_field, database_info)
                if matched_field:
                    self.query_parts['order_by'].append((matched_field, direction))



#%%
def find_common_columns(table1: str, table2: str, database_info: Dict) -> str:
    """Find columns with the same name in both tables for joining."""
    if table1 not in database_info or table2 not in database_info:
        return None
        
    # Get columns from both tables
    table1_cols = set(database_info[table1]['columns'])
    table2_cols = set(database_info[table2]['columns'])
    
    # Find common columns
    common_cols = table1_cols.intersection(table2_cols)
    
    if common_cols:
        # Use the first common column found
        join_col = next(iter(common_cols))
        return f"{table1}.{join_col} = {table2}.{join_col}"
        
    return None



#%%
def get_sql_database_info(engine) -> Dict[str, Dict[str, List[str]]]:
    """
    Retrieve database schema information with fixed column handling.
    """
    try:
        inspector = inspect(engine)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        
        database_info = {}
        
        # Get all tables
        for table_name in inspector.get_table_names():
            # Initialize table info
            database_info[table_name] = {
                'columns': [],
                'foreign_keys': [],
                'primary_key': [],
                'relationships': [],
            }
            
            # Get columns - don't create variations, just use actual columns
            columns = inspector.get_columns(table_name)
            for column in columns:
                database_info[table_name]['columns'].append(column['name'])
                
            # Get primary key
            pk = inspector.get_pk_constraint(table_name)
            if pk and 'constrained_columns' in pk:
                database_info[table_name]['primary_key'].extend(pk['constrained_columns'])
                
            # Get foreign keys
            fks = inspector.get_foreign_keys(table_name)
            for fk in fks:
                for col_name, ref_col in zip(fk['constrained_columns'], fk['referred_columns']):
                    database_info[table_name]['foreign_keys'].append(
                        (col_name, fk['referred_table'], ref_col)
                    )
                    if fk['referred_table'] not in database_info[table_name]['relationships']:
                        database_info[table_name]['relationships'].append(fk['referred_table'])
            
            # Add table variations (just singular/plural)
            table_variations = [table_name]
            if table_name.endswith('s'):
                table_variations.append(table_name[:-1])
            database_info[table_name]['variations'] = list(set(table_variations))
        
        return database_info
        
    except Exception as e:
        print(f"Error getting database info: {str(e)}")
        return {}

def print_database_info(database_info: Dict) -> None:
    """
    Print the database schema information in a readable format.
    Useful for debugging and verification.
    """
    print("\nDatabase Schema Information:")
    print("=" * 80)
    
    for table_name, table_info in database_info.items():
        print(f"\nTable: {table_name}")
        print("-" * 40)
        
        print("\nColumns:")
        for column in table_info['columns']:
            pk_marker = "*" if column in table_info['primary_key'] else " "
            print(f"  {pk_marker} {column}")
            
        if table_info['foreign_keys']:
            print("\nForeign Keys:")
            for fk in table_info['foreign_keys']:
                print(f"  {fk[0]} -> {fk[1]}.{fk[2]}")
            
        if table_info['relationships']:
            print("\nRelationships:")
            for rel in table_info['relationships']:
                print(f"  - {rel}")
                
        if table_info.get('variations'):
            print("\nName Variations:")
            print(f"  {', '.join(table_info['variations'])}")
            
        print("-" * 40)
        
def generate_sql_query(message: str, database_info: Dict) -> str:
    """Generate a SQL query from the recognized patterns."""
    recognizer = QueryRecognizer()
    query_parts = recognizer.recognize_patterns(message, database_info)
    
    # Build the SQL query
    sql_parts = []
    
    # SELECT clause
    if query_parts['select']:
        # If we have both group by and aggregations, add group by columns to select
        if query_parts['group_by'] and any('COUNT(' in s or 'AVG(' in s or 'SUM(' in s or 'MIN(' in s or 'MAX(' in s for s in query_parts['select']):
            select_items = query_parts['group_by'].copy()
            select_items.extend(query_parts['select'])
            select_clause = "SELECT " + ", ".join(select_items)
        else:
            select_clause = "SELECT " + ", ".join(query_parts['select'])
    else:
        select_clause = "SELECT *"
    
    sql_parts.append(select_clause)
    
    # FROM clause
    if query_parts['from']:
        sql_parts.append(f"FROM {query_parts['from'][0]}")
    
    # JOIN clauses
    for join_table in query_parts['join']:
        join_condition = find_common_columns(query_parts['from'][0], join_table, database_info)
        if join_condition:
            sql_parts.append(f"JOIN {join_table} ON {join_condition}")
            
    # WHERE clause
    if query_parts['where'] and not(query_parts['group_by']) :
        sql_parts.append("WHERE " + " AND ".join(query_parts['where']))
    
    # GROUP BY clause 
    if query_parts['group_by']:
        sql_parts.append("GROUP BY " + ", ".join(query_parts['group_by']))
    
    # HAVING clause
    if query_parts['having']:
        sql_parts.append("HAVING " + " AND ".join(query_parts['having']))
    
    # ORDER BY clause
    if query_parts['order_by']:
        order_clauses = [f"{field} {direction}" for field, direction in query_parts['order_by']]
        sql_parts.append("ORDER BY " + ", ".join(order_clauses))
    
    return " ".join(sql_parts)



#%%
from sqlalchemy import create_engine
import pandas as pd
import os
from typing import Optional, Tuple
import sys
from colorama import init, Fore, Style
from tabulate import tabulate
import time

# Initialize colorama for cross-platform colored output
init()

def setup_database_connection() -> Optional[create_engine]:
    """Setup database connection with error handling."""
    try:
        MYSQL_HOST = 'localhost'
        MYSQL_USER = 'root'
        MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '00112000')
        MYSQL_DB = 'mysql_551_enrollment'
        MYSQL_PORT = 3306
        engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")
        print(f"{Fore.GREEN}[✓] Database connection successful{Style.RESET_ALL}")
        return engine
    except Exception as e:
        print(f"{Fore.RED}[✗] Database connection failed: {str(e)}{Style.RESET_ALL}")
        return None

def print_results(query: str, results: Optional[pd.DataFrame]) -> None:
    """Print query results in a formatted way."""
    print("\n" + "="*80)
    print(f"{Fore.CYAN}Executed Query:{Style.RESET_ALL}")
    print("-"*80)
    print(query)
    print("-"*80)
    
    if results is not None and not results.empty:
        print(f"\n{Fore.GREEN}Results:{Style.RESET_ALL}")
        print(tabulate(results.head(10), headers='keys', tablefmt='pretty', showindex=False))
        if len(results) > 10:
            print(f"\n{Fore.YELLOW}Showing first 10 of {len(results)} results{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}No results found{Style.RESET_ALL}")

def interactive_mode(engine) -> None:
    """Run the program in interactive mode."""
    print(f"\n{Fore.CYAN}=== Interactive SQL Query Generator ==={Style.RESET_ALL}")
    print("Type 'exit' to quit, 'test' to run test queries, or enter your query in natural language.")
    
    while True:
        try:
            print("\n" + "-"*80)
            user_input = input(f"{Fore.GREEN}Enter your query:{Style.RESET_ALL} ").strip()
            
            if user_input.lower() == 'exit':
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break
            elif user_input.lower() == 'test':
                run_test_queries(engine)
                continue
                
            if not user_input:
                print(f"{Fore.YELLOW}Please enter a query{Style.RESET_ALL}")
                continue
                
            start_time = time.time()
            database_info = get_sql_database_info(engine)
            query = generate_sql_query(user_input, database_info)
            
            # Execute query
            results = pd.read_sql(query, engine)
            execution_time = time.time() - start_time
            
            print_results(query, results)
            print(f"\n{Fore.CYAN}Query executed in {execution_time:.2f} seconds{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def run_test_queries(engine) -> None:
    """Run a set of predefined test queries."""
    test_queries = [
        # Basic queries
        "show all students",
        "get all courses",
        "list departments",
        
        # Filtered queries
        "show students where gpa is greater than 3.5",
        "get students where major contains 'Computer'",
        
        # Join queries
        "show enrollments with student details",
        "display courses with instructor details",
        "list grades with student and course details",
        
        # Aggregation queries
        "count students per major",
        
        # Complex queries
        "display count of students per department where gpa greater than 3.5 order by count descending",
        "get courses with credits greater than 3 and department contains 'Science' order by credits"
    ]
    
    print(f"\n{Fore.CYAN}Running test queries...{Style.RESET_ALL}\n")
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n{Fore.CYAN}Test Query #{i}:{Style.RESET_ALL} {query}")
            database_info = get_sql_database_info(engine)
            sql_query = generate_sql_query(query, database_info)
            results = pd.read_sql(sql_query, engine)
            print_results(sql_query, results)
            
            if i < len(test_queries):
                input(f"\n{Fore.YELLOW}Press Enter for next query...{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error executing test query #{i}: {str(e)}{Style.RESET_ALL}")
            if i < len(test_queries):
                input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")

def main():
    """Main program entry point."""
    try:
        # Setup database connection
        engine = setup_database_connection()
        if not engine:
            sys.exit(1)
            
        # Run interactive mode
        interactive_mode(engine)
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Program terminated by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
    finally:
        print("\n" + "="*80)

if __name__ == "__main__":
    main()