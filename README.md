# MyLath

A high-performance graph database with vector search capabilities, built on Redis and Python.

## Features

- **Graph Operations**: Create, read, update, and delete nodes and edges
- **Graph Traversals**: Powerful traversal API similar to Gremlin
- **Vector Search**: Fast similarity search with HNSW indexing
- **Secondary Indices**: Efficient property-based lookups
- **REST API**: Complete HTTP API for all operations
- **Redis Backend**: Leverages Redis for high performance and scalability

## Quick Start

### Installation

```bash
pip install mylath
```

### Basic Usage

```python
from mylath import Graph, RedisStorage

# Initialize
storage = RedisStorage(host='localhost', port=6379)
graph = Graph(storage)

# Create nodes
alice = graph.create_node("person", {"name": "Alice", "age": 30})
bob = graph.create_node("person", {"name": "Bob", "age": 25})

# Create relationships
friendship = graph.create_edge("knows", alice.id, bob.id, {"since": "2020"})

# Graph traversal
friends = graph.V(alice.id).out("knows").to_list()
print([f.properties["name"] for f in friends])  # ['Bob']

# Vector operations
vector = graph.vectors.add_vector(
    [0.1, 0.2, 0.3, 0.4], 
    metadata={"type": "embedding"}
)

similar = graph.vectors.search_vectors([0.15, 0.25, 0.35, 0.45], k=5)
```

### Running the API Server

```python
from mylath.api import GraphAPI
from mylath import RedisStorage

storage = RedisStorage()
api = GraphAPI(storage)
api.run(host='0.0.0.0', port=5000)
```

Or use the CLI:

```bash
mylath-server --host 0.0.0.0 --port 5000
```

## API Reference

### Node Operations

```bash
# Create node
curl -X POST http://localhost:5000/nodes \
  -H "Content-Type: application/json" \
  -d '{"label": "person", "properties": {"name": "Alice", "age": 30}}'

# Get node
curl http://localhost:5000/nodes/{node_id}

# Update node
curl -X PUT http://localhost:5000/nodes/{node_id} \
  -H "Content-Type: application/json" \
  -d '{"properties": {"age": 31}}'

# Delete node
curl -X DELETE http://localhost:5000/nodes/{node_id}
```

### Edge Operations

```bash
# Create edge
curl -X POST http://localhost:5000/edges \
  -H "Content-Type: application/json" \
  -d '{"label": "knows", "from_node": "node1_id", "to_node": "node2_id", "properties": {"since": "2020"}}'

# Get edge
curl http://localhost:5000/edges/{edge_id}

# Delete edge
curl -X DELETE http://localhost:5000/edges/{edge_id}
```

### Vector Operations

```bash
# Add vector
curl -X POST http://localhost:5000/vectors \
  -H "Content-Type: application/json" \
  -d '{"data": [0.1, 0.2, 0.3, 0.4], "metadata": {"type": "embedding"}}'

# Search vectors
curl -X POST http://localhost:5000/vectors/search \
  -H "Content-Type: application/json" \
  -d '{"query_vector": [0.15, 0.25, 0.35, 0.45], "k": 5, "metric": "cosine"}'
```

### Query Operations

```bash
# Execute traversal query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "type": "traversal",
    "params": {
      "steps": [
        {"type": "V", "params": {"node_ids": ["node1_id"]}},
        {"type": "out", "params": {"label": "knows"}},
        {"type": "has", "params": {"key": "age", "value": 25}},
        {"type": "limit", "params": {"count": 10}}
      ]
    }
  }'
```

## Graph Traversal API

MyLath provides a powerful traversal API similar to Apache TinkerPop Gremlin:

```python
# Basic traversals
graph.V().has("label", "person")  # All person nodes
graph.V(node_id).out("knows")     # Friends of a person
graph.V(node_id).in_("knows")     # People who know this person
graph.V(node_id).both("knows")    # All connected people

# Chaining operations
graph.V().has("age", 30).out("works_at").in_("works_at").dedup()

# Path finding
path = graph.V(start_id).shortest_path(end_id, "knows")

# Filtering and limiting
graph.V().has("label", "person").filter(lambda n: n.properties["age"] > 25).limit(10)

# Counting
count = graph.V().has("label", "person").count()

# Getting property values
names = graph.V().has("label", "person").values("name")
```

## Configuration

```python
from mylath.config import MyLathConfig

config = MyLathConfig(
    redis_host='localhost',
    redis_port=6379,
    redis_db=0,
    vector_m=16,                    # HNSW max connections
    vector_ef_construction=128,     # HNSW construction parameter
    vector_ef_search=768,           # HNSW search parameter
    api_host='0.0.0.0',
    api_port=5000
)
```

## Performance Considerations

### Redis Configuration

For optimal performance, configure Redis with:

```redis
# redis.conf
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Indexing Strategy

- Use secondary indices for frequently queried properties
- Consider property value distribution when creating indices
- Monitor index usage with graph statistics

### Vector Search Optimization

- Tune HNSW parameters based on your data:
  - `m`: Higher values = better recall, more memory
  - `ef_construction`: Higher values = better index quality, slower indexing
  - `ef_search`: Higher values = better recall, slower search

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Graph API     │    │  Vector API     │    │   Query API     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Graph Core    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Storage Layer   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    └─────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.




















