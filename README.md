# Cypher Planner 🚀

A sophisticated Cypher query parser and logical execution plan generator designed for hybrid Redis + GraphBLAS architectures.

## 🎯 Overview

This system converts Cypher queries into optimized logical execution plans that leverage:
- **Redis** for fast property storage and retrieval
- **GraphBLAS** for high-performance graph traversals using sparse matrix operations

## 🏗️ Architecture

```
Cypher Query
     ↓
AST Parser
     ↓
Logical Planner
     ↓
Rule-Based Optimizer
     ↓
Physical Planner
     ↓
Redis + GraphBLAS Operations
```

## ⚡ Quick Start

### Installation

1. **Clone or create the project structure:**
```bash
mkdir cypher_planner
cd cypher_planner
```

2. **Set up the project files** (see Project Structure below)

3. **Run the system:**
```bash
python main.py
```

### Basic Usage

```python
from cypher_planner.parser import CypherParser
from cypher_planner.logical_planner import LogicalPlanner
from cypher_planner.optimizer import RuleBasedOptimizer

# Initialize components
parser = CypherParser()
planner = LogicalPlanner()
optimizer = RuleBasedOptimizer()

# Parse and plan a query
query = "MATCH (u:User {country: 'USA'})-[:FOLLOWS]->(f:User) RETURN u.name, f.name"
ast = parser.parse(query)
logical_plan = planner.create_logical_plan(ast)
optimized_plan = optimizer.optimize(logical_plan)
```

## 📁 Project Structure

```
cypher_planner/
├── README.md
├── requirements.txt
├── setup.py
├── main.py                     # Main entry point
├── cypher_planner/
│   ├── __init__.py
│   ├── ast_nodes.py           # AST node definitions
│   ├── parser.py              # Cypher parser
│   ├── logical_operators.py   # Logical plan operators
│   ├── logical_planner.py     # Logical plan generation
│   ├── optimizer.py           # Rule-based optimizer
│   ├── physical_planner.py    # Physical plan generation
│   ├── cost_model.py          # Cost estimation
│   └── utils.py               # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_planner.py
│   └── test_optimizer.py
└── examples/
    ├── basic_queries.py
    └── complex_queries.py
```

## 🔧 Features

### Supported Cypher Features
- ✅ MATCH patterns with nodes and relationships
- ✅ WHERE clauses with property and structural filters
- ✅ RETURN projections with aliases
- ✅ ORDER BY, SKIP, LIMIT
- ✅ Variable-length paths (`*1..3`)
- ✅ OPTIONAL MATCH (basic support)
- ✅ Property filters on nodes and relationships

### Optimization Features
- ✅ **Predicate Pushdown**: Moves filters to scan operations
- ✅ **Filter Combining**: Merges adjacent filter operations
- ✅ **Execution Target Optimization**: Routes operations to Redis vs GraphBLAS
- ✅ **Scan Optimization**: Prioritizes selective operations

### Physical Planning
- ✅ **Redis Operations**: SMEMBERS, HGET, SCAN commands
- ✅ **GraphBLAS Operations**: Matrix-vector and matrix-matrix multiplications
- ✅ **Hybrid Coordination**: Manages data flow between systems

## 📊 Example Output

**Query:**
```cypher
MATCH (u:User {country: 'USA'})-[:FOLLOWS]->(f:User) 
WHERE u.age > 25 
RETURN u.name, f.name 
ORDER BY u.name LIMIT 10
```

**Logical Plan:**
```
Limit(skip=0, limit=10)
  OrderBy([(u.name, True)])
    Project(u.name, f.name)
      Filter((u.age > 25))
        Expand(u)-[:FOLLOWS]->(f)
          NodeScan(u:User {'country': 'USA'})
```

**Physical Plan:**
```
[Redis] Limit
  [Redis] OrderBy
    [Redis] Project
      > HGET node:{id} name
    [Redis] PropertyFilter
      [GraphBLAS] Expand
        > v_f = v_u @ A_FOLLOWS
      [Redis] NodeScan
        > SMEMBERS label:User
        > SMEMBERS prop:country:USA
```

## 🚀 Running the System

### Method 1: Test Mode
```bash
python main.py
# Choose option 1 for test queries
```

### Method 2: Interactive Mode
```bash
python main.py
# Choose option 2 for interactive testing
# Then enter your own Cypher queries
```

### VSCode Setup

1. **Open VSCode in the project directory:**
```bash
code .
```

2. **Install Python extension** if not already installed

3. **Set Python interpreter:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Choose your Python 3.8+ interpreter

4. **Run the main file:**
   - Open `main.py`
   - Press `F5` or click the "Run" button
   - Or use terminal: `python main.py`

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=cypher_planner

# Run specific test file
python -m pytest tests/test_parser.py -v
```

## 🔮 Future Enhancements

### Planned Features
- [ ] Cost-based optimization with statistics
- [ ] Support for aggregations (COUNT, SUM, COLLECT)
- [ ] Write operations (CREATE, SET, DELETE, MERGE)
- [ ] Subqueries and UNION operations
- [ ] Advanced join algorithms
- [ ] Query caching and plan reuse

### Integration Opportunities
- [ ] Actual Redis backend integration
- [ ] GraphBLAS library bindings
- [ ] Performance benchmarking
- [ ] Query execution engine
- [ ] Statistics collection system

## 📝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCypher project for Cypher language specification
- GraphBLAS community for sparse matrix computing standards
- Redis team for excellent key-value storage