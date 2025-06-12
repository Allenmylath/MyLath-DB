# STEP 5: Create new file cypher_planner/execution_statistics.py

"""
Execution Statistics for Cost-Based Optimization
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class ExecutionStatistics:
    """Statistics for cost-based optimization"""
    node_count: int = 1000000
    edge_count: int = 5000000
    label_cardinalities: Dict[str, int] = field(default_factory=dict)
    property_selectivity: Dict[str, float] = field(default_factory=dict)
    edge_type_cardinalities: Dict[str, int] = field(default_factory=dict)
    index_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.label_cardinalities:
            # Default label cardinalities
            self.label_cardinalities = {
                'Person': 500000,
                'User': 300000,
                'Actor': 10000,
                'Movie': 50000,
                'Product': 100000,
                'Car': 75000,
                'Tweet': 1000000,
                'Director': 5000
            }
        
        if not self.property_selectivity:
            # Default property selectivity estimates
            self.property_selectivity = {
                'Person.age': 0.02,
                'User.country': 0.1,
                'Product.price': 0.05,
                'Car.make': 0.15,
                'Movie.year': 0.05,
                'Tweet.hashtags': 0.3,
                'Actor.name': 0.0001,
                'Person.name': 0.0001
            }
        
        if not self.edge_type_cardinalities:
            # Default edge type cardinalities
            self.edge_type_cardinalities = {
                'FOLLOWS': 2000000,
                'ACTED_IN': 100000,
                'DIRECTED': 20000,
                'POSTED': 5000000,
                'OWNS': 500000,
                'FRIENDS_WITH': 1000000,
                'KNOWS': 3000000
            }
        
        if not self.index_statistics:
            # Default index information
            self.index_statistics = {
                'label_indexes': {
                    'Person': {'selectivity': 0.5, 'cost_factor': 1.0},
                    'User': {'selectivity': 0.3, 'cost_factor': 1.0},
                    'Actor': {'selectivity': 0.01, 'cost_factor': 0.5},
                    'Movie': {'selectivity': 0.05, 'cost_factor': 0.7}
                },
                'property_indexes': {
                    'age': {'selectivity': 0.02, 'cost_factor': 0.3},
                    'country': {'selectivity': 0.1, 'cost_factor': 0.4},
                    'name': {'selectivity': 0.0001, 'cost_factor': 0.2}
                }
            }
    
    def get_label_cardinality(self, label: str) -> int:
        """Get cardinality estimate for a label"""
        return self.label_cardinalities.get(label, self.node_count // 20)
    
    def get_property_selectivity(self, variable: str, property_key: str) -> float:
        """Get selectivity estimate for a property filter"""
        key = f"{variable}.{property_key}"
        if key in self.property_selectivity:
            return self.property_selectivity[key]
        
        # Fallback to generic property selectivity
        generic_key = property_key
        return self.property_selectivity.get(generic_key, 0.1)
    
    def get_edge_cardinality(self, edge_type: str) -> int:
        """Get cardinality estimate for an edge type"""
        return self.edge_type_cardinalities.get(edge_type, self.edge_count // 10)
    
    def has_label_index(self, label: str) -> bool:
        """Check if label has an index"""
        return label in self.index_statistics.get('label_indexes', {})
    
    def has_property_index(self, property_key: str) -> bool:
        """Check if property has an index"""
        return property_key in self.index_statistics.get('property_indexes', {})
    
    def get_index_cost_factor(self, index_type: str, key: str) -> float:
        """Get cost factor for using an index"""
        indexes = self.index_statistics.get(f"{index_type}_indexes", {})
        return indexes.get(key, {}).get('cost_factor', 1.0)

class StatisticsCollector:
    """Collects and maintains execution statistics"""
    
    def __init__(self):
        self.statistics = ExecutionStatistics()
        self.query_history = []
        self.performance_metrics = {}
    
    def update_cardinality_estimate(self, label: str, actual_count: int):
        """Update cardinality estimate based on actual results"""
        self.statistics.label_cardinalities[label] = actual_count
    
    def update_selectivity_estimate(self, property_key: str, actual_selectivity: float):
        """Update selectivity estimate based on actual results"""
        self.statistics.property_selectivity[property_key] = actual_selectivity
    
    def record_query_performance(self, query: str, execution_time: float, cardinality: int):
        """Record query performance for future optimization"""
        self.query_history.append({
            'query': query,
            'execution_time': execution_time,
            'cardinality': cardinality,
            'timestamp': 'now'  # In real implementation, use actual timestamp
        })
    
    def get_statistics(self) -> ExecutionStatistics:
        """Get current statistics"""
        return self.statistics