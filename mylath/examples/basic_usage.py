# examples/basic_usage.py
"""
Basic usage example for MyLath
"""

from mylath import Graph, RedisStorage

def main():
    # Initialize
    storage = RedisStorage(host='localhost', port=6379, db=0)
    graph = Graph(storage)
    
    print("MyLath Basic Usage Example")
    print("=" * 30)
    
    # Create some nodes
    print("\n1. Creating nodes...")
    alice = graph.create_node("person", {"name": "Alice", "age": 30, "city": "NYC"})
    bob = graph.create_node("person", {"name": "Bob", "age": 25, "city": "SF"})
    charlie = graph.create_node("person", {"name": "Charlie", "age": 35, "city": "LA"})
    
    # Create a company
    company = graph.create_node("company", {"name": "TechCorp", "industry": "Software"})
    
    print(f"Created nodes: {alice.id}, {bob.id}, {charlie.id}, {company.id}")
    
    # Create relationships
    print("\n2. Creating relationships...")
    friendship1 = graph.create_edge("knows", alice.id, bob.id, {"since": "2020", "closeness": 8})
    friendship2 = graph.create_edge("knows", bob.id, charlie.id, {"since": "2019", "closeness": 6})
    
    employment1 = graph.create_edge("works_at", alice.id, company.id, {"role": "Engineer", "salary": 120000})
    employment2 = graph.create_edge("works_at", bob.id, company.id, {"role": "Designer", "salary": 100000})
    
    print(f"Created edges: {friendship1.id}, {friendship2.id}, {employment1.id}, {employment2.id}")
    
    # Graph traversals
    print("\n3. Graph traversals...")
    
    # Find Alice's friends
    alice_friends = graph.V(alice.id).out("knows").to_list()
    print(f"Alice's friends: {[f.properties['name'] for f in alice_friends]}")
    
    # Find people who work at TechCorp
    employees = graph.V(company.id).in_("works_at").to_list()
    print(f"TechCorp employees: {[e.properties['name'] for e in employees]}")
    
    # Find Alice's colleagues (people who work at same company)
    colleagues = (graph.V(alice.id)
                      .out("works_at")
                      .in_("works_at")
                      .to_list())
    colleague_names = [c.properties['name'] for c in colleagues if c.id != alice.id]
    print(f"Alice's colleagues: {colleague_names}")
    
    # Find path from Alice to Charlie
    path = graph.V(alice.id).shortest_path(charlie.id, "knows")
    if path:
        path_names = [p.properties['name'] for p in path]
        print(f"Path from Alice to Charlie: {' -> '.join(path_names)}")
    
    # Property-based queries
    print("\n4. Property-based queries...")
    
    # Find people over 30
    older_people = graph.V().has("age").filter(lambda n: n.properties.get("age", 0) > 30).to_list()
    print(f"People over 30: {[p.properties['name'] for p in older_people]}")
    
    # Find people in NYC
    nyc_people = graph.find_nodes_by_property("city", "NYC")
    print(f"People in NYC: {[p.properties['name'] for p in nyc_people]}")
    
    # Vector operations
    print("\n5. Vector operations...")
    
    # Add some document embeddings
    doc1_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    doc2_embedding = [0.2, 0.3, 0.4, 0.5, 0.6]
    doc3_embedding = [0.8, 0.1, 0.9, 0.2, 0.1]
    
    vec1 = graph.vectors.add_vector(doc1_embedding, 
                                   metadata={"type": "document"},
                                   properties={"title": "AI Research Paper", "author": "Alice"})
    
    vec2 = graph.vectors.add_vector(doc2_embedding,
                                   metadata={"type": "document"}, 
                                   properties={"title": "ML Tutorial", "author": "Bob"})
    
    vec3 = graph.vectors.add_vector(doc3_embedding,
                                   metadata={"type": "image"},
                                   properties={"title": "Cat Photo", "tags": ["animal", "cute"]})
    
    print(f"Added vectors: {vec1.id}, {vec2.id}, {vec3.id}")
    
    # Search for similar documents
    query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]
    similar_docs = graph.vectors.search_vectors(query_embedding, k=2, 
                                               filters={"type": "document"})
    
    print("Similar documents:")
    for doc, score in similar_docs:
        print(f"  - {doc.properties['title']} (score: {score:.3f})")
    
    # Graph statistics
    print("\n6. Graph statistics...")
    stats = graph.get_stats()
    print(f"Graph stats: {stats}")
    
    print("\nExample completed successfully! 🎉")


if __name__ == "__main__":
    main()