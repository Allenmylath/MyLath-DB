# example_house_search_1000.py
"""
Enhanced example demonstrating hybrid search for houses with 1000 nodes:
1. Create 1000 houses with properties and embeddings
2. Measure ingestion time and search performance
3. Search by property filters + vector similarity
"""

import random
import numpy as np
import time
from typing import List, Dict, Any
from mylath.mylath.storage.redis_storage import RedisStorage
from mylath.mylath.graph.graph import Graph


def generate_dummy_embedding(text: str, dim: int = 128) -> list:
    """Generate dummy embedding based on text (for demo purposes)"""
    # Simple hash-based embedding for demo
    random.seed(hash(text) % (2**31))
    return [random.uniform(-1, 1) for _ in range(dim)]


def generate_house_data(num_houses: int = 1000) -> List[Dict[str, Any]]:
    """Generate realistic house data for testing"""
    
    locations = ["downtown", "suburbs", "uptown", "midtown", "riverside", "hillside", "lakeside", "parkside"]
    house_types = ["house", "apartment", "condo", "townhouse", "duplex"]
    
    # Description templates for variety
    description_templates = [
        "{adjective} {type} with {feature1} and {feature2}",
        "{feature1} {type} in {location_desc} with {feature2}",
        "Beautiful {type} featuring {feature1}, {feature2}, and {feature3}",
        "{adjective} {type} with {feature1}, perfect for {lifestyle}",
        "Spacious {type} offering {feature1} and {feature2} in {location_desc}"
    ]
    
    adjectives = ["Modern", "Cozy", "Luxury", "Charming", "Spacious", "Updated", "Beautiful", "Elegant", "Contemporary", "Classic"]
    features = [
        "large backyard", "updated kitchen", "city views", "rooftop garden", "pool access",
        "garage", "fireplace", "hardwood floors", "granite counters", "walk-in closets",
        "balcony", "patio", "basement", "attic", "bay windows", "crown molding",
        "stainless appliances", "marble bathrooms", "vaulted ceilings", "garden space"
    ]
    location_descriptions = [
        "quiet neighborhood", "bustling area", "peaceful surroundings", "convenient location",
        "tree-lined street", "family-friendly area", "vibrant community", "sought-after district"
    ]
    lifestyles = ["families", "professionals", "retirees", "students", "young couples"]
    
    houses_data = []
    
    for i in range(num_houses):
        # Random property values
        bedrooms = random.choice([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5])  # Weighted towards 2-3 BR
        bathrooms = random.choice([1, 1, 2, 2, 2, 3, 3, 4])  # Weighted towards 2-3 BA
        location = random.choice(locations)
        house_type = random.choice(house_types)
        
        # Price based on bedrooms, bathrooms, and location with some randomness
        base_price = 200000 + (bedrooms * 50000) + (bathrooms * 30000)
        if location in ["downtown", "uptown"]:
            base_price *= 1.3
        elif location in ["riverside", "lakeside"]:
            base_price *= 1.2
        
        # Add randomness ±20%
        price = int(base_price * random.uniform(0.8, 1.2))
        
        # Generate description
        template = random.choice(description_templates)
        selected_features = random.sample(features, min(3, len(features)))
        
        description = template.format(
            adjective=random.choice(adjectives),
            type=house_type,
            feature1=selected_features[0] if len(selected_features) > 0 else "great features",
            feature2=selected_features[1] if len(selected_features) > 1 else "excellent condition",
            feature3=selected_features[2] if len(selected_features) > 2 else "premium finishes",
            location_desc=random.choice(location_descriptions),
            lifestyle=random.choice(lifestyles)
        )
        
        house_data = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "details": description,
            "price": price,
            "location": location,
            "type": house_type,
            "house_id": f"house_{i+1:04d}"  # Add unique identifier
        }
        
        houses_data.append(house_data)
    
    return houses_data


def create_houses_with_timing(graph: Graph, houses_data: List[Dict[str, Any]]) -> tuple:
    """Create houses and measure ingestion time"""
    
    print(f"\n📊 INGESTION PERFORMANCE TEST")
    print(f"Creating {len(houses_data)} house nodes...")
    print("-" * 60)
    
    start_time = time.time()
    batch_start = start_time
    created_houses = []
    batch_size = 100
    
    for i, house_data in enumerate(houses_data):
        # Generate embedding for details
        details_embedding = generate_dummy_embedding(house_data["details"])
        
        # Create node with properties and embedding
        house = graph.create_node(
            label="house",
            properties=house_data,
            embeddings={
                "details_emb": details_embedding
            }
        )
        
        created_houses.append(house)
        
        # Progress reporting every batch_size houses
        if (i + 1) % batch_size == 0:
            batch_end = time.time()
            batch_time = batch_end - batch_start
            total_time = batch_end - start_time
            rate = batch_size / batch_time
            
            print(f"  ✓ Created {i+1:4d}/{len(houses_data)} houses | "
                  f"Batch: {batch_time:.2f}s ({rate:.1f} houses/sec) | "
                  f"Total: {total_time:.2f}s")
            
            batch_start = batch_end
    
    end_time = time.time()
    total_ingestion_time = end_time - start_time
    ingestion_rate = len(houses_data) / total_ingestion_time
    
    print(f"\n✅ INGESTION COMPLETE")
    print(f"   Total time: {total_ingestion_time:.2f} seconds")
    print(f"   Average rate: {ingestion_rate:.1f} houses per second")
    print(f"   Time per house: {total_ingestion_time/len(houses_data)*1000:.2f} ms")
    
    return created_houses, total_ingestion_time


def timed_search(search_name: str, search_func) -> tuple:
    """Execute a search function and measure its execution time"""
    
    print(f"\n🔍 {search_name}")
    print("-" * 60)
    
    start_time = time.time()
    results = search_func()
    end_time = time.time()
    
    search_time = end_time - start_time
    
    print(f"Search completed in {search_time*1000:.2f} ms")
    print(f"Found {len(results)} results")
    
    return results, search_time


def comprehensive_search_tests(graph: Graph) -> Dict[str, float]:
    """Run comprehensive search tests and measure performance"""
    
    print("\n" + "="*80)
    print("🚀 SEARCH PERFORMANCE TESTS")
    print("="*80)
    
    search_times = {}
    
    # Test 1: Simple property filter
    def property_search():
        return (graph.V()
                .has_label("house")
                .has_properties(bedrooms=3, bathrooms=2)
                .to_results())
    
    results, search_time = timed_search("Property Filter: 3BR/2BA", property_search)
    search_times["property_filter"] = search_time
    
    # Show sample results
    for i, result in enumerate(results[:3]):
        props = result['properties']
        print(f"  {i+1}. {props['bedrooms']}BR/{props['bathrooms']}BA - ${props['price']:,} - {props['location']}")
    if len(results) > 3:
        print(f"  ... and {len(results)-3} more")
    
    
    # Test 2: Price range filter
    def price_range_search():
        return (graph.V()
                .has_label("house")
                .has_properties(price_min=300000, price_max=500000)
                .to_results())
    
    results, search_time = timed_search("Price Range: $300k-500k", price_range_search)
    search_times["price_range"] = search_time
    
    print(f"  Price range: ${min(r['properties']['price'] for r in results):,} - ${max(r['properties']['price'] for r in results):,}")
    
    
    # Test 3: Vector search only
    def vector_search():
        query_embedding = generate_dummy_embedding("modern apartment with city views")
        return (graph.V()
                .has_label("house")
                .vector_search("details_emb", query_embedding, k=20)
                .to_results())
    
    results, search_time = timed_search("Vector Search: 'modern apartment with city views' (k=20)", vector_search)
    search_times["vector_search"] = search_time
    
    for i, result in enumerate(results[:3]):
        props = result['properties']
        similarity = result['vector_similarity']
        print(f"  {i+1}. Similarity: {similarity:.3f} - {props['type']} - ${props['price']:,}")
        print(f"     {props['details'][:60]}...")
    
    
    # Test 4: Hybrid search (most complex)
    def hybrid_search():
        query_embedding = generate_dummy_embedding("luxury house with garden and garage")
        return (graph.V()
                .has_label("house")
                .has_properties(bedrooms_min=2, price_max=600000, type=["house", "townhouse"])
                .vector_search("details_emb", query_embedding, k=15)
                .to_results())
    
    results, search_time = timed_search("Hybrid Search: Properties + Vector similarity", hybrid_search)
    search_times["hybrid_search"] = search_time
    
    for i, result in enumerate(results[:3]):
        props = result['properties']
        similarity = result['vector_similarity']
        print(f"  {i+1}. Similarity: {similarity:.3f} - {props['bedrooms']}BR - ${props['price']:,}")
        print(f"     {props['details'][:60]}...")
    
    
    # Test 5: Large k vector search
    def large_k_search():
        query_embedding = generate_dummy_embedding("family friendly neighborhood")
        return (graph.V()
                .has_label("house")
                .vector_search("details_emb", query_embedding, k=100)
                .to_results())
    
    results, search_time = timed_search("Large Vector Search: k=100", large_k_search)
    search_times["large_vector_search"] = search_time
    
    print(f"  Retrieved top {len(results)} most similar houses")
    
    return search_times


def print_performance_summary(ingestion_time: float, search_times: Dict[str, float], num_houses: int):
    """Print comprehensive performance summary"""
    
    print("\n" + "="*80)
    print("📈 PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n🏠 DATA INGESTION ({num_houses} houses):")
    print(f"   Total time: {ingestion_time:.2f}s")
    print(f"   Rate: {num_houses/ingestion_time:.1f} houses/sec")
    print(f"   Time per house: {ingestion_time/num_houses*1000:.2f}ms")
    
    print(f"\n🔍 SEARCH PERFORMANCE:")
    for search_type, time_taken in search_times.items():
        print(f"   {search_type.replace('_', ' ').title():<25}: {time_taken*1000:>6.2f}ms")
    
    print(f"\n📊 SEARCH STATISTICS:")
    avg_search_time = sum(search_times.values()) / len(search_times)
    print(f"   Average search time: {avg_search_time*1000:.2f}ms")
    print(f"   Fastest search: {min(search_times.values())*1000:.2f}ms ({min(search_times, key=search_times.get)})")
    print(f"   Slowest search: {max(search_times.values())*1000:.2f}ms ({max(search_times, key=search_times.get)})")


def flush_existing_data(graph: Graph):
    """Flush all existing data before starting the test"""
    
    print("🧹 FLUSHING EXISTING DATA")
    print("-" * 40)
    
    try:
        # Get current stats before flushing
        stats = graph.get_stats()
        current_nodes = stats.get('nodes', 0)
        current_vectors = stats.get('redis_official_vectors', 0)
        
        if current_nodes > 0 or current_vectors > 0:
            print(f"  Found existing data:")
            print(f"    Nodes: {current_nodes}")
            print(f"    Vectors: {current_vectors}")
            print("  Clearing all data...")
            
            # Flush the Redis database
            graph.storage.client.flushdb()
            
            # Recreate the vector index
            graph.storage._ensure_vector_index()
            
            print("  ✅ Data cleared successfully")
        else:
            print("  ✅ No existing data found")
            
    except Exception as e:
        print(f"  ⚠️  Warning: Could not clear existing data: {e}")
        print("  Continuing with existing data...")
    
    print()


def main():
    """Main function to run the enhanced example"""
    
    print("🏠 MyLath Enhanced House Search Demo - 25,000 Nodes")
    print("=" * 60)
    
    # Initialize MyLath
    storage = RedisStorage()
    graph = Graph(storage)
    
    # Flush existing data
    flush_existing_data(graph)
    
    # Generate house data
    print("\n📋 Generating house data...")
    houses_data = generate_house_data(25000)
    print(f"Generated {len(houses_data)} house records")
    
    # Sample of generated data
    print("\n📝 Sample house data:")
    for i in range(3):
        house = houses_data[i]
        print(f"  {house['house_id']}: {house['bedrooms']}BR/{house['bathrooms']}BA - "
              f"${house['price']:,} - {house['location']} {house['type']}")
        print(f"    {house['details']}")
    print("  ...")
    
    # Create houses with timing
    houses, ingestion_time = create_houses_with_timing(graph, houses_data)
    
    # Run comprehensive search tests
    search_times = comprehensive_search_tests(graph)
    
    # Show graph statistics
    print("\n" + "="*80)
    print("📊 GRAPH STATISTICS")
    print("="*80)
    stats = graph.get_stats()
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key.capitalize()}: {value:,}")
        else:
            print(f"  {key.capitalize()}: {value}")
    
    # Performance summary
    print_performance_summary(ingestion_time, search_times, len(houses_data))
    
    print(f"\n✅ Demo completed! Created {len(houses)} houses and measured performance.")
    print(f"   Total runtime: {ingestion_time + sum(search_times.values()):.2f}s")


if __name__ == "__main__":
    main()