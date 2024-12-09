# ChatToDB - Find Your Data
### Group 37 - Natural Language Database Query Interface

ChatDB is an intelligent database interface that allows users to query both SQL and NoSQL databases using natural language. The system supports pattern-based query generation for both MySQL and MongoDB, with built-in data upload capabilities and example query generation.

## Features

- Natural language to SQL/MongoDB query conversion
- Support for both MySQL and MongoDB databases
- Pattern-based query generation system
- Fuzzy matching for schema identification
- Data upload functionality (CSV/XLSX for MySQL, JSON for MongoDB)
- Example query generation
- Real-time query execution
- Interactive web interface

## Tech Stack

### Backend
- Python 3.x
- Flask (Web Framework)
- SQLAlchemy (SQL ORM)
- PyMongo (MongoDB Client)
- Pandas (Data Processing)
- FuzzyWuzzy (String Matching)

### Frontend
- HTML5/CSS3
- Bootstrap
- JavaScript
- Socket.io

### Databases
- MySQL
- MongoDB

## Installation

1. Clone the repository
```bash
git clone [your-repository-url]
cd chatdb
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure database connections in `config.py`:
```python
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',
    'port': 3306
}

MONGODB_URI = "your_mongodb_connection_string"
```

## Project Structure
```
chatdb/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── models.py
│   └── utils/
│       ├── sql_generator.py
│       ├── mongo_generator.py
│       └── pattern_matcher.py
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── config.py
├── requirements.txt
└── run.py
```

## Query Pattern Examples

### SQL Patterns
```python
# WHERE clause pattern
"Show me all orders where total is greater than 100"
→ SELECT * FROM orders WHERE total > 100

# GROUP BY pattern
"Show total sales by product category"
→ SELECT category, SUM(sales) FROM products GROUP BY category

# JOIN pattern
"Show customer names and their orders"
→ SELECT customers.name, orders.* FROM customers 
   JOIN orders ON customers.id = orders.customer_id
```

### MongoDB Patterns
```python
# Match pattern
"Find all products with price less than 50"
→ db.products.find({"price": {"$lt": 50}})

# Aggregation pattern
"Show average rating by product category"
→ db.products.aggregate([
    {"$group": {"_id": "$category", "avg_rating": {"$avg": "$rating"}}}
])
```

## Usage

1. Start the application:
```bash
python run.py
```

2. Access the web interface at `http://localhost:5000`

3. Select database type (MySQL/MongoDB)

4. Upload your data:
   - MySQL: CSV or XLSX files
   - MongoDB: JSON files

5. Use natural language to query your data:
   - View example queries
   - Enter your own queries
   - See generated query syntax
   - View query results

## Key Features

### Pattern Matching System
- Template-based query generation
- Fuzzy matching for schema identification
- Support for common query patterns:
  - WHERE conditions
  - GROUP BY aggregations
  - JOIN operations
  - Complex filtering

### Data Upload
- CSV/XLSX support for MySQL
- JSON support for MongoDB
- Automatic schema detection
- Data preview functionality

### Query Generation
- Natural language processing
- Pattern recognition
- Schema validation
- Query optimization

## Limitations
- Fixed pattern recognition system
- Limited support for aliases
- Basic query logic only
- No support for complex subqueries

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
- Course instructors and TAs for guidance
- Open source community for various libraries used
- Database community for inspiration and support
