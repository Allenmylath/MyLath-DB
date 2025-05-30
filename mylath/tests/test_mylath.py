













# Example usage and tests
# test_mylath.py
def test_mylath():
    """Test basic MyLath functionality"""
    import time
    
    # Initialize storage and graph
    storage = RedisStorage()
    graph = Graph(storage)
    
    # Test node operations
    print("Testing node operations...")
    person1 = graph.create_node("person", {"name": "Alice", "age": 30})
    person2 = graph.create_node("person", {"name": "Bob", "age": 25})
    company = graph.create_node("company", {"name": "TechCorp"})
    
    print(f"Created nodes: {person1.id}, {person2.id}, {company.id}")
    
    # Test edge operations
    print("Testing edge operations...")
    friendship = graph.create_edge("knows", person1.id, person2.id, 
                                  {"since": "2020"})
    employment1 = graph.create_edge("works_at", person1.id, company.id,
                                   {"role": "Engineer"})
    employment2 = graph.create_edge("works_at", person2.id, company.id,
                                   {"role": "Designer"})
    
    print(f"Created edges: {friendship.id}, {employment1.id}, {employment2.id}")
    
    # Test traversals
    print("Testing traversals...")
    
    # Find Alice's friends
    friends = graph.V(person1.id).out("knows").to_list()
    print(f"Alice's friends: {[f.properties['name'] for f in friends]}")
    
    # Find people who work at TechCorp
    employees = graph.V(company.id).in_("works_at").to_list()
    print(f"TechCorp employees: {[e.properties['name'] for e in employees]}")
    
    # Find Alice's colleagues (people who work at the same company)
    colleagues = (graph.V(person1.id)
                      .out("works_at")
                      .in_("works_at")
                      .has("name")
                      .to_list())
    print(f"Alice's colleagues: {[c.properties.get('name') for c in colleagues]}")
    
    # Test vector operations
    print("Testing vector operations...")
    
    # Add some sample vectors
    vec1 = graph.vectors.add_vector([0.1, 0.2, 0.3, 0.4], 
                                   metadata={"type": "document"},
                                   properties={"source": "doc1"})
    vec2 = graph.vectors.add_vector([0.2, 0.3, 0.4, 0.5],
                                   metadata={"type": "document"}, 
                                   properties={"source": "doc2"})
    vec3 = graph.vectors.add_vector([0.8, 0.9, 0.1, 0.2],
                                   metadata={"type": "image"},
                                   properties={"source": "img1"})
    
    print(f"Created vectors: {vec1.id}, {vec2.id}, {vec3.id}")
    
    # Search for similar vectors
    query_vec = [0.15, 0.25, 0.35, 0.45]
    similar_vecs = graph.vectors.search_vectors(query_vec, k=2)
    print(f"Similar vectors: {[(v.id, score) for v, score in similar_vecs]}")
    
    # Test with filters
    doc_vecs = graph.vectors.search_vectors(
        query_vec, k=2, 
        filters={"type": "document"}
    )
    print(f"Similar document vectors: {[(v.id, score) for v, score in doc_vecs]}")
    
    # Get stats
    stats = graph.get_stats()
    print(f"Graph stats: {stats}")
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    test_mylath()