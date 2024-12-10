# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 00:00:16 2024

@author: Administrator
"""

import random
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId


def extract_mongodb_schema_full(database):
    """
    Extract schema information from a MongoDB database.

    Args:
        database: MongoDB database object.

    Returns:
        dict: A dictionary containing collection names and their schema details.
    """
    schema_info = {}
    for collection_name in database.list_collection_names():
        collection = database[collection_name]
        sample_documents = collection.find().limit(3)  # Fetch a sample of documents

        # Infer schema from sample documents
        inferred_schema = {}
        for document in sample_documents:
            for key, value in document.items():
                value_type = type(value).__name__
                if key not in inferred_schema:
                    inferred_schema[key] = value_type
                elif inferred_schema[key] != value_type:
                    inferred_schema[key] = "Mixed"

        schema_info[collection_name] = inferred_schema

    return schema_info


def print_database_schema(schema_info):
    """
    Print the schema information for a MongoDB database.

    Args:
        schema_info (dict): Schema information for the database.
    """
    print("\nDatabase Schema:")
    for collection, schema in schema_info.items():
        print(f"Collection: {collection}")
        for field, field_type in schema.items():
            print(f"  {field}: {field_type}")
        print()


def generate_query_examples(schema_info):
    """
    Generate example MongoDB queries based on schema information.

    Args:
        schema_info (dict): Schema information for the database.

    Returns:
        List[str]: A list of example MongoDB queries as strings.
    """
    examples = []

    # Generate find query examples
    for collection, schema in schema_info.items():
        if schema:
            field = random.choice(list(schema.keys()))
            value_type = schema[field]
            if value_type == "str":
                example = f"db.{collection}.find({{{field}: 'example_value'}})"
            elif value_type == "int":
                example = f"db.{collection}.find({{{field}: {{'$gt': 10}}}})"
            else:
                example = f"db.{collection}.find({{{field}: 'value_based_on_type'}})"
            examples.append(example)

    # Generate aggregation query examples
    for collection in schema_info.keys():
        examples.append(
            f"db.{collection}.aggregate([{{'$group': {{'_id': '$field', 'count': {{'$sum': 1}}}}}}])"
        )

    return examples

def get_mongo_example_with_match(schema_info):
    """
    Generate a natural language example query for MongoDB with a structured match condition.
    
    Args:
        schema_info (dict): Dictionary containing collection names and their schema details.
    
    Returns:
        str: A natural language example query for MongoDB.
    """
    import random

    try:
        # Randomly select an action keyword
        action = random.choice(["find", "retrieve", "get", "fetch", "list", "display"])

        # Flatten all fields across collections and their types
        all_fields = []
        for collection, fields in schema_info.items():
            for field_name, field_type in fields.items():
                all_fields.append((collection, field_name, field_type))

        # Randomly choose between numeric and string-based conditions
        condition_type = random.choice(["number", "string"])

        if condition_type == "number":
            # Filter fields of numeric type
            numeric_fields = [
                (collection, field) for collection, field, field_type in all_fields if field_type in ["int", "float"]
            ]
            if not numeric_fields:
                raise ValueError("No numeric fields available for generating a match condition.")

            collection, field = random.choice(numeric_fields)
            condition_keyword = random.choice(["greater than", "less than", "equals", "at least", "at most"])
            condition_value = random.randint(10, 100)  # Example numeric condition

            condition = f"{field} {condition_keyword} {condition_value}"

        else:  # String-based condition
            # Filter fields of string type
            string_fields = [
                (collection, field) for collection, field, field_type in all_fields if field_type == "str"
            ]
            if not string_fields:
                raise ValueError("No string fields available for generating a match condition.")

            collection, field = random.choice(string_fields)
            condition_keyword = random.choice(["is", "equals", "contains", "starts with", "ends with"])
            condition_value = "example_value"  # Example string value

            if condition_keyword == "contains":
                condition = f"{field} {condition_keyword} '{condition_value}'"
            elif condition_keyword in ["starts with", "ends with"]:
                condition = f"{field} {condition_keyword} '{condition_value}'"
            else:
                condition = f"{field} {condition_keyword} '{condition_value}'"

        # Construct the natural language example
        connector = random.choice(["that", "with"])
        suggested_message = f"{action} documents from the {collection} collection {connector} {condition}."

        return suggested_message
    except Exception as e:
        print(f"[ERROR] Error in get_example_with_match: {str(e)}")
        return "Could not generate an example query."



import random
import re


def get_mongo_example_with_lookup(schema_info):
    """
    Generate a natural language example query for MongoDB with a structured lookup condition.
    
    Args:
        schema_info (dict): Dictionary containing collection names and their schema details.
    
    Returns:
        str: A natural language example query for MongoDB involving a lookup operation.
    """
    try:
        # Randomly select a collection to act as the primary
        primary_collection = random.choice(list(schema_info.keys()))

        # Find potential foreign collections for lookup
        foreign_collections = []
        for foreign_collection, foreign_fields in schema_info.items():
            if foreign_collection != primary_collection:
                for field_name, field_type in foreign_fields.items():
                    # Look for ObjectId or other joinable types
                    if field_type == "ObjectId":
                        foreign_collections.append((foreign_collection, field_name))
                        break

        if not foreign_collections:
            return "No collections suitable for generating a lookup example."

        # Randomly select a foreign collection and its field
        foreign_collection, foreign_field = random.choice(foreign_collections)

        # Generate a random lookup phrase based on patterns
        lookup_phrases = [
            f"join with {foreign_collection}",
            f"with {foreign_collection} details",
            f"include {foreign_collection}",
            f"related {foreign_collection}",
        ]
        lookup_phrase = random.choice(lookup_phrases)

        # Construct the natural language example query
        suggested_message = (
            f"Get documents from the {primary_collection} collection {lookup_phrase}."
        )

        return suggested_message

    except Exception as e:
        print(f"[ERROR] Error in generate_mongo_query_with_lookup: {str(e)}")
        return "Could not generate an example query."




def get_mongo_example_with_group_by(schema_info):
    """
    Generate a natural language example query for MongoDB with a structured group by condition.
    
    Args:
        schema_info (dict): Dictionary containing collection names and their schema details.
    
    Returns:
        str: A natural language example query for MongoDB involving a group by operation.
    """
    import random

    try:
        # Randomly select a collection to act as the primary
        primary_collection = random.choice(list(schema_info.keys()))

        # Flatten all fields across the primary collection
        collection_fields = schema_info[primary_collection]

        # Randomly choose a field for grouping
        group_fields = [
            field for field, field_type in collection_fields.items() if field_type in ["str", "int", "ObjectId"]
        ]
        if not group_fields:
            return f"No fields available in the {primary_collection} collection for grouping."

        group_field = random.choice(group_fields)

        # Randomly choose an aggregation operation
        agg_operations = ["count", "sum", "average", "min", "max"]
        agg_op = random.choice(agg_operations)

        # Select a numeric field for aggregation if applicable
        numeric_fields = [
            field for field, field_type in collection_fields.items() if field_type in ["int", "float"]
        ]
        if numeric_fields and agg_op != "count":
            agg_field = random.choice(numeric_fields)
            aggregation_phrase = f"{agg_op} of {agg_field}"
        else:
            agg_field = None
            aggregation_phrase = agg_op

        # Generate random group-by phrasing
        group_phrases = [
            f"per {group_field}",
            f"in each {group_field}",
            f"for each {group_field}",
        ]
        group_phrase = random.choice(group_phrases)

        # Construct the natural language query
        suggested_message = (
            f"Get the {aggregation_phrase} from the {primary_collection} collection {group_phrase}."
        )

        return suggested_message

    except Exception as e:
        print(f"[ERROR] Error in generate_mongo_query_with_group_by: {str(e)}")
        return "Could not generate an example query."


def main():
    """
    Main function to test MongoDB schema extraction and query generation.
    """
    # Replace with your MongoDB connection details
    MONGODB_URI = "mongodb+srv://shilongr:001120rsl@cluster0.pzrgm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    db = client['mydata551']

    # Step 1: Extract schema information
    print("Extracting database schema...")
    schema_info = extract_mongodb_schema_full(db)
    print_database_schema(schema_info)
    
    

    # Step 2: Generate example queries
    print("\nGenerating example queries...")
    
    query_examples = get_mongo_example_with_match(schema_info)
    print(query_examples)
    
    
    query_examples = get_mongo_example_with_lookup(schema_info)
    print(query_examples)
    
    
    query_examples = get_mongo_example_with_group_by(schema_info)
    print(query_examples)


# Run the main function
if __name__ == "__main__":
    main()
