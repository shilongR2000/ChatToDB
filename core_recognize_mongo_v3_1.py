# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:12:03 2024

@author: Administrator
"""

import re
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import process
from datetime import datetime
import json
from pymongo import MongoClient
from typing import Dict, List, Set
from collections import defaultdict
from pymongo.server_api import ServerApi
from bson import ObjectId
import pandas as pd
from bson import json_util

class MongoDBQueryRecognizer:
    def __init__(self):
        # Initialize patterns for MongoDB aggregation
        self.agg_patterns = {
            'average': {'$avg': ''},
            'avg': {'$avg': ''},
            'min': {'$min': ''},
            'max': {'$max': ''},
            'sum': {'$sum': ''},
            'total': {'$sum': ''},
            'count': {'$sum': 1}
        }
        
        self.match_patterns = {
            r'([\w_]+)\s+is\s+([\w\s]+)': lambda f, v: {f: v.strip()},
            r'([\w_]+)\s+equals?\s+([\w\s]+)': lambda f, v: {f: v.strip()},
            r'([\w_]+)\s+greater\s+than\s+(\d+(?:\.\d+)?)': lambda f, v: {f: {'$gt': float(v)}},
            r'([\w_]+)\s+more\s+than\s+(\d+(?:\.\d+)?)': lambda f, v: {f: {'$gt': float(v)}},
            r'([\w_]+)\s+less\s+than\s+(\d+(?:\.\d+)?)': lambda f, v: {f: {'$lt': float(v)}},
            r'([\w_]+)\s+under\s+(\d+(?:\.\d+)?)': lambda f, v: {f: {'$lt': float(v)}},
            r'([\w_]+)\s+contains?\s+([\w\s]+)': lambda f, v: {f: {'$regex': v.strip(), '$options': 'i'}},
            r'([\w_]+)\s+starts?\s+with\s+([\w\s]+)': lambda f, v: {f: {'$regex': f'^{v.strip()}', '$options': 'i'}},
            r'([\w_]+)\s+ends?\s+with\s+([\w\s]+)': lambda f, v: {f: {'$regex': f'{v.strip()}$', '$options': 'i'}}
        }
        
        self.group_patterns = [
            r'\bper\b\s+([\w_]+)',
            r'\bin each\b\s+([\w_]+)',
            r'\bfor each\b\s+([\w_]+)'
        ]
        
        self.lookup_patterns = [
            r'with\s+([\w_]+)\s+details?',
            r'join\s+with\s+([\w_]+)',
            r'include\s+([\w_]+)',
            r'related\s+([\w_]+)'
        ]
        
        self.sort_patterns = [
            (r'order\s+by\s+([\w_]+)\s+desc(?:ending)?', -1),
            (r'order\s+by\s+([\w_]+)\s+asc(?:ending)?', 1),
            (r'sort\s+by\s+([\w_]+)\s+desc(?:ending)?', -1),
            (r'sort\s+by\s+([\w_]+)\s+asc(?:ending)?', 1),
            (r'order\s+by\s+([\w_]+)', 1),
            (r'sort\s+by\s+([\w_]+)', 1)
        ]

        self.pipeline_stages = []

    def _fuzzy_match_field(self, field_name: str, collection_info: Dict, threshold: int = 80) -> Optional[str]:
        """Fuzzy match a field name against available fields."""
        field_matches = process.extract(
            field_name,
            collection_info['fields'],
            limit=1
        )
        
        if field_matches and field_matches[0][1] >= threshold:
            return field_matches[0][0]
        return None

    def _fuzzy_match_collection(self, collection_name: str, database_info: Dict, threshold: int = 80) -> Optional[str]:
        """Fuzzy match a collection name against available collections."""
        collection_matches = process.extract(
            collection_name,
            list(database_info.keys()),
            limit=1
        )
        
        if collection_matches and collection_matches[0][1] >= threshold:
            return collection_matches[0][0]
        return None

    def recognize_patterns(self, message: str, database_info: Dict, main_collection: str) -> List[Dict]:
        """Recognize patterns and convert to MongoDB aggregation pipeline stages."""
        self.pipeline_stages = []
        
        # Recognize match conditions
        self._recognize_match_conditions(message, database_info[main_collection])
        
        # Recognize lookups (joins)
        self._recognize_lookups(message, database_info)
        
        # Recognize group operations
        self._recognize_group_operations(message, database_info[main_collection])
        
        # Recognize sort operations
        self._recognize_sort_operations(message, database_info[main_collection])
        
        return self.pipeline_stages

    def _recognize_match_conditions(self, message: str, collection_info: Dict):
        """Convert where conditions to $match stage."""
        match_conditions = {}
        
        for pattern, formatter in self.match_patterns.items():
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                field = match.group(1)
                value = match.group(2)
                
                matched_field = self._fuzzy_match_field(field, collection_info)
                if matched_field:
                    try:
                        condition = formatter(matched_field, value)
                        match_conditions.update(condition)
                    except Exception as e:
                        print(f"Error formatting condition: {str(e)}")
                        continue
        
        if match_conditions:
            self.pipeline_stages.append({'$match': match_conditions})

    def _recognize_lookups(self, message: str, database_info: Dict):
        """Convert joins to $lookup stages."""
        for pattern in self.lookup_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                foreign_collection = match.group(1)
                matched_collection = self._fuzzy_match_collection(foreign_collection, database_info)
                
                if matched_collection:
                    lookup_stage = {
                        '$lookup': {
                            'from': matched_collection,
                            'localField': '_id',  # Default to _id, can be customized
                            'foreignField': f'{matched_collection.lower()}_id',
                            'as': f'{matched_collection.lower()}_details'
                        }
                    }
                    self.pipeline_stages.append(lookup_stage)
                    
                    # Add unwind stage to flatten the array
                    self.pipeline_stages.append({
                        '$unwind': {
                            'path': f'${matched_collection.lower()}_details',
                            'preserveNullAndEmptyArrays': True
                        }
                    })

    def _recognize_group_operations(self, message: str, collection_info: Dict):
        """Convert group by and aggregations to $group stage."""
        group_fields = []
        group_operations = {}
        
        # Check for group by fields
        for pattern in self.group_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                field = match.group(1)
                matched_field = self._fuzzy_match_field(field, collection_info)
                if matched_field:
                    group_fields.append(matched_field)
        
        # Also check for direct field mentions in "per field" pattern
        words = message.lower().split()
        if 'per' in words:
            idx = words.index('per')
            if idx + 1 < len(words):
                field = words[idx + 1]
                matched_field = self._fuzzy_match_field(field, collection_info)
                if matched_field and matched_field not in group_fields:
                    group_fields.append(matched_field)
        
        # Check for aggregation operations
        words = message.lower().split()
        for agg_word, agg_op in self.agg_patterns.items():
            if agg_word in words:
                idx = words.index(agg_word)
                if idx + 1 < len(words):
                    field = words[idx + 1]
                    matched_field = self._fuzzy_match_field(field, collection_info)
                    if matched_field:
                        if isinstance(agg_op, dict):
                            op_key = list(agg_op.keys())[0]
                            op_value = agg_op[list(agg_op.keys())[0]]
                            if op_value == '':
                                op_value = f'${matched_field}'
                            group_operations[f'{agg_word}_{matched_field}'] = {op_key: op_value}
                else:
                    # Handle count without specific field
                    if agg_word == 'count':
                        group_operations['count'] = {'$sum': 1}
        
        if group_fields or group_operations:
            group_stage = {'$group': {}}
            
            # Add group by fields
            if group_fields:
                if len(group_fields) == 1:
                    # For single field, use simpler _id structure
                    group_stage['$group']['_id'] = f'${group_fields[0]}'
                else:
                    # For multiple fields, use object structure
                    group_stage['$group']['_id'] = {
                        field: f'${field}' for field in group_fields
                    }
            else:
                group_stage['$group']['_id'] = None
            
            # Add aggregation operations
            if not group_operations and group_fields:
                # If no explicit aggregation but we have group fields, add count
                group_operations['count'] = {'$sum': 1}
            
            group_stage['$group'].update(group_operations)
            
            self.pipeline_stages.append(group_stage)
            
            # Add sort stage for count queries
            if 'count' in group_operations:
                self.pipeline_stages.append({'$sort': {'count': -1}})
                
            
    def _recognize_sort_operations(self, message: str, collection_info: Dict):
        """Convert order by to $sort stage."""
        sort_fields = {}
        
        for pattern, direction in self.sort_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                field = match.group(1)
                matched_field = self._fuzzy_match_field(field, collection_info)
                if matched_field:
                    if group_operations := any('$group' in stage for stage in self.pipeline_stages):
                        # If we have a group stage, sort by the aggregation result
                        for stage in self.pipeline_stages:
                            if '$group' in stage:
                                for key in stage['$group'].keys():
                                    if key != '_id' and matched_field in key:
                                        sort_fields[key] = direction
                    else:
                        sort_fields[matched_field] = direction
        
        if sort_fields:
            self.pipeline_stages.append({'$sort': sort_fields})


def extract_mongodb_schema(db) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract schema information from MongoDB database including:
    - Collection names
    - Fields in each collection
    - Relationships between collections based on field names and references
    
    Args:
        db: MongoDB database connection
    
    Returns:
        Dict containing schema information for each collection
    """
    database_info = {}
    
    # Get all collection names
    collections = db.list_collection_names()
    
    # Helper function to extract field names from a document
    def extract_fields(document: Dict, prefix: str = '') -> Set[str]:
        fields = set()
        for key, value in document.items():
            field_name = f"{prefix}{key}"
            fields.add(field_name)
            
            # Handle nested documents
            if isinstance(value, dict):
                nested_fields = extract_fields(value, f"{field_name}.")
                fields.update(nested_fields)
        return fields

    # Helper function to identify relationships
    def identify_relationships(field_name: str, collections: List[str]) -> List[str]:
        relationships = []
        # Convert to singular form for matching
        field_base = field_name.replace('_id', '').replace('_ids', '')
        field_singular = field_base[:-1] if field_base.endswith('s') else field_base
        
        for collection in collections:
            # Convert collection name to singular form
            coll_singular = collection[:-1] if collection.endswith('s') else collection
            
            # Check if field matches collection name (both plural and singular forms)
            if (field_base == collection or 
                field_base == coll_singular or 
                field_singular == coll_singular):
                relationships.append(collection)
        
        return relationships

    # Process each collection
    for collection_name in collections:
        # Initialize collection info
        database_info[collection_name] = {
            'fields': set(),
            'relationships': set(),
            'sample_values': defaultdict(set)
        }
        
        # Sample documents to identify fields and their types
        sample_docs = list(db[collection_name].aggregate([
            {'$sample': {'size': 100}},  # Sample up to 100 documents
            {'$limit': 100}
        ]))
        
        # Process each sample document
        for doc in sample_docs:
            # Extract fields
            fields = extract_fields(doc)
            database_info[collection_name]['fields'].update(fields)
            
            # Store sample values for each field (useful for relationship detection)
            for field in fields:
                value = doc.get(field)
                if value is not None:
                    database_info[collection_name]['sample_values'][field].add(str(type(value).__name__))

    # Second pass to identify relationships
    for collection_name, info in database_info.items():
        # Look for fields that might indicate relationships
        for field in info['fields']:
            # Check for _id fields or array fields that might be references
            if '_id' in field or any(t in info['sample_values'][field] for t in ['ObjectId', 'list']):
                relationships = identify_relationships(field, collections)
                info['relationships'].update(relationships)
        
        # Convert sets to sorted lists for consistency
        info['fields'] = sorted(list(info['fields']))
        info['relationships'] = sorted(list(info['relationships']))
        del info['sample_values']  # Remove temporary sample values

    return database_info

def print_database_schema(database_info: Dict) -> None:
    """
    Print the extracted MongoDB schema information in a readable format.
    """
    print("\nMongoDB Schema Information:")
    print("=" * 80)
    
    for collection_name, collection_info in database_info.items():
        print(f"\nCollection: {collection_name}")
        print("-" * 40)
        
        print("\nFields:")
        for field in collection_info['fields']:
            print(f"  - {field}")
            
        if collection_info['relationships']:
            print("\nRelationships:")
            for relationship in collection_info['relationships']:
                print(f"  - {relationship}")
        
        print("-" * 40)


def detect_main_collection(message: str, schema_info: Dict) -> Optional[str]:
    """
    Detect the main collection from the natural language query using only schema information.
    Returns the collection name or None if no collection is detected.
    """
    message = message.lower()
    collection_scores = {coll: 0 for coll in schema_info.keys()}
    
    for collection, info in schema_info.items():
        # Check for collection name mention (both plural and singular forms)
        singular = collection[:-1] if collection.endswith('s') else collection
        if collection in message or singular in message:
            collection_scores[collection] += 5
            
        # Check for field names from the collection
        for field in info['fields']:
            # Split nested field names (e.g., "address.city" -> ["address", "city"])
            field_parts = field.lower().split('.')
            for part in field_parts:
                if part in message:
                    collection_scores[collection] += 2
        
        # Check for relationship mentions
        if 'relationships' in info:
            for relationship in info['relationships']:
                if relationship.lower() in message:
                    collection_scores[collection] += 1

    # Get collection with highest score
    if collection_scores:
        max_score = max(collection_scores.values())
        if max_score > 0:
            return max(collection_scores.items(), key=lambda x: x[1])[0]
    
    return None

def generate_mongodb_query(message: str, schema_info: Dict) -> Tuple[str, List[Dict]]:
    """
    Generate a MongoDB aggregation pipeline from natural language.
    Returns a tuple of (collection_name, pipeline).
    """
    # Detect main collection
    main_collection = detect_main_collection(message, schema_info)
    if not main_collection:
        raise ValueError("Could not determine which collection to query from the message")
    
    # Generate pipeline
    recognizer = MongoDBQueryRecognizer()
    pipeline = recognizer.recognize_patterns(message, schema_info, main_collection)
    
    # If no pipeline stages, return a simple find all pipeline
    if not pipeline:
        pipeline = [{'$match': {}}]
    
    return main_collection, pipeline

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, ObjectId)):
            return str(obj)
        return super().default(obj)

# Example usage:
if __name__ == "__main__":
    
    MONGODB_URI = "mongodb+srv://shilongr:001120rsl@cluster0.pzrgm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    db = client['mydata551']
    
    
    # Extract schema information
    schema_info = extract_mongodb_schema(db)
    
    # Print schema information
    print_database_schema(schema_info)
    

    # Test queries
    test_queries = [
        "show all brands",
        "get documents from the orders collection with totalAmount equals 67.",
        "get products where price is greater than 100",
        "show products where stock less than 10",
        "count orders per status",
        "show reviews with products details",
        "Get the sum from the categories collection for each name."
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        try:
            # Generate pipeline
            collection, pipeline = generate_mongodb_query(query, schema_info)
            print(f"\nCollection: {collection}")
            print("MongoDB Pipeline:")
            print(json.dumps(pipeline, indent=2))
            
            # Execute pipeline and show results
            # results = list(db[collection].aggregate(pipeline))
            result = list(db[collection].aggregate(pipeline))
            result_df = pd.json_normalize(json.loads(json_util.dumps(result)))
            print(result_df)
            # print("\nResults (first 5):")
            # for result in results[:5]:
            #     print(json.dumps(result, indent=2, cls=MongoJSONEncoder))
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print('='*50)