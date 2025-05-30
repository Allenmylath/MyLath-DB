# mylath/graph/aggregation.py
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from collections import defaultdict, Counter
import statistics
import math
from dataclasses import dataclass
from enum import Enum


class AggregateFunction(Enum):
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STDDEV = "stddev"
    VARIANCE = "variance"
    COLLECT = "collect"
    COLLECT_SET = "collect_set"  # unique values only
    FIRST = "first"
    LAST = "last"
    PERCENTILE = "percentile"


@dataclass
class AggregateResult:
    """Result of an aggregation operation"""
    groups: Dict[Any, Dict[str, Any]]
    total_groups: int
    total_items: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "groups": self.groups,
            "total_groups": self.total_groups,
            "total_items": self.total_items
        }


class AggregationEngine:
    """Core aggregation engine for MyLath"""
    
    def __init__(self):
        self.aggregators = {
            AggregateFunction.COUNT: self._count,
            AggregateFunction.SUM: self._sum,
            AggregateFunction.AVG: self._avg,
            AggregateFunction.MIN: self._min,
            AggregateFunction.MAX: self._max,
            AggregateFunction.MEDIAN: self._median,
            AggregateFunction.STDDEV: self._stddev,
            AggregateFunction.VARIANCE: self._variance,
            AggregateFunction.COLLECT: self._collect,
            AggregateFunction.COLLECT_SET: self._collect_set,
            AggregateFunction.FIRST: self._first,
            AggregateFunction.LAST: self._last,
            AggregateFunction.PERCENTILE: self._percentile,
        }
    
    def aggregate(self, items: List[Any], 
                 group_by: Optional[Union[str, List[str], Callable]] = None,
                 aggregations: Dict[str, Union[str, Tuple[str, Any]]] = None,
                 having: Optional[Callable] = None,
                 order_by: Optional[Union[str, List[str], Callable]] = None,
                 limit: Optional[int] = None) -> AggregateResult:
        """
        Perform aggregation on items
        
        Args:
            items: List of nodes/edges to aggregate
            group_by: Property name(s) or function to group by
            aggregations: Dict of {result_name: (function, property)} or {result_name: function}
            having: Filter function for groups
            order_by: Property or function to order results
            limit: Maximum number of groups to return
        """
        if not items:
            return AggregateResult({}, 0, 0)
        
        if aggregations is None:
            aggregations = {"count": AggregateFunction.COUNT}
            
        # Group items
        groups = self._group_items(items, group_by)
        
        # Apply aggregations to each group
        result_groups = {}
        for group_key, group_items in groups.items():
            group_result = {}
            for result_name, agg_spec in aggregations.items():
                if isinstance(agg_spec, tuple):
                    func, property_name = agg_spec
                    if isinstance(property_name, dict):  # For percentile: ("percentile", {"property": "age", "percentile": 0.95})
                        prop = property_name.get("property")
                        extra_params = {k: v for k, v in property_name.items() if k != "property"}
                    else:
                        prop = property_name
                        extra_params = {}
                else:
                    func = agg_spec
                    prop = None
                    extra_params = {}
                
                if isinstance(func, str):
                    func = AggregateFunction(func)
                
                value = self.aggregators[func](group_items, prop, **extra_params)
                group_result[result_name] = value
            
            result_groups[group_key] = group_result
        
        # Apply HAVING filter
        if having:
            result_groups = {k: v for k, v in result_groups.items() if having(v)}
        
        # Order results
        if order_by:
            result_groups = self._order_groups(result_groups, order_by)
        
        # Apply limit
        if limit and len(result_groups) > limit:
            result_groups = dict(list(result_groups.items())[:limit])
        
        return AggregateResult(
            groups=result_groups,
            total_groups=len(result_groups),
            total_items=len(items)
        )
    
    def _group_items(self, items: List[Any], group_by: Optional[Union[str, List[str], Callable]]) -> Dict[Any, List[Any]]:
        """Group items by specified criteria"""
        if group_by is None:
            return {"_all": items}
        
        groups = defaultdict(list)
        
        for item in items:
            if callable(group_by):
                key = group_by(item)
            elif isinstance(group_by, list):
                # Multiple properties
                key = tuple(item.properties.get(prop) for prop in group_by)
            else:
                # Single property
                key = item.properties.get(group_by)
            
            groups[key].append(item)
        
        return dict(groups)
    
    def _order_groups(self, groups: Dict[Any, Dict[str, Any]], order_by: Union[str, Callable]) -> Dict[Any, Dict[str, Any]]:
        """Order groups by specified criteria"""
        if callable(order_by):
            key_func = lambda item: order_by(item[1])  # item[1] is the group result dict
        else:
            key_func = lambda item: item[1].get(order_by, 0)  # order by aggregation result
        
        sorted_items = sorted(groups.items(), key=key_func, reverse=True)
        return dict(sorted_items)
    
    # Aggregation functions
    def _count(self, items: List[Any], property_name: str = None, **kwargs) -> int:
        if property_name:
            return sum(1 for item in items if property_name in item.properties)
        return len(items)
    
    def _sum(self, items: List[Any], property_name: str, **kwargs) -> float:
        values = [item.properties.get(property_name, 0) for item in items 
                 if property_name in item.properties]
        return sum(v for v in values if isinstance(v, (int, float)))
    
    def _avg(self, items: List[Any], property_name: str, **kwargs) -> float:
        values = [item.properties.get(property_name, 0) for item in items 
                 if property_name in item.properties]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        return statistics.mean(numeric_values) if numeric_values else 0
    
    def _min(self, items: List[Any], property_name: str, **kwargs) -> Any:
        values = [item.properties.get(property_name) for item in items 
                 if property_name in item.properties]
        return min(values) if values else None
    
    def _max(self, items: List[Any], property_name: str, **kwargs) -> Any:
        values = [item.properties.get(property_name) for item in items 
                 if property_name in item.properties]
        return max(values) if values else None
    
    def _median(self, items: List[Any], property_name: str, **kwargs) -> float:
        values = [item.properties.get(property_name, 0) for item in items 
                 if property_name in item.properties]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        return statistics.median(numeric_values) if numeric_values else 0
    
    def _stddev(self, items: List[Any], property_name: str, **kwargs) -> float:
        values = [item.properties.get(property_name, 0) for item in items 
                 if property_name in item.properties]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        return statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
    
    def _variance(self, items: List[Any], property_name: str, **kwargs) -> float:
        values = [item.properties.get(property_name, 0) for item in items 
                 if property_name in item.properties]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        return statistics.variance(numeric_values) if len(numeric_values) > 1 else 0
    
    def _collect(self, items: List[Any], property_name: str, **kwargs) -> List[Any]:
        return [item.properties.get(property_name) for item in items 
                if property_name in item.properties]
    
    def _collect_set(self, items: List[Any], property_name: str, **kwargs) -> List[Any]:
        values = [item.properties.get(property_name) for item in items 
                 if property_name in item.properties]
        return list(set(values))
    
    def _first(self, items: List[Any], property_name: str, **kwargs) -> Any:
        for item in items:
            if property_name in item.properties:
                return item.properties[property_name]
        return None
    
    def _last(self, items: List[Any], property_name: str, **kwargs) -> Any:
        for item in reversed(items):
            if property_name in item.properties:
                return item.properties[property_name]
        return None
    
    def _percentile(self, items: List[Any], property_name: str, percentile: float = 0.5, **kwargs) -> float:
        values = [item.properties.get(property_name, 0) for item in items 
                 if property_name in item.properties]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if not numeric_values:
            return 0
        sorted_values = sorted(numeric_values)
        k = (len(sorted_values) - 1) * percentile
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


# mylath/graph/query_builder.py
from typing import List, Dict, Any, Optional, Union, Callable
from ..storage.redis_storage import RedisStorage, Node, Edge
from .traversal import GraphTraversal
from .aggregation import AggregationEngine, AggregateResult


class QueryBuilder:
    """Advanced query builder with aggregation support"""
    
    def __init__(self, storage: RedisStorage):
        self.storage = storage
        self.aggregation_engine = AggregationEngine()
        self._traversal = None
        self._group_by = None
        self._aggregations = {}
        self._having = None
        self._order_by = None
        self._limit = None
    
    def match(self, traversal: GraphTraversal) -> 'QueryBuilder':
        """Set the base traversal"""
        self._traversal = traversal
        return self
    
    def group_by(self, *properties: str) -> 'QueryBuilder':
        """Group results by properties"""
        if len(properties) == 1:
            self._group_by = properties[0]
        else:
            self._group_by = list(properties)
        return self
    
    def group_by_func(self, func: Callable) -> 'QueryBuilder':
        """Group by custom function"""
        self._group_by = func
        return self
    
    def count(self, alias: str = "count", property_name: str = None) -> 'QueryBuilder':
        """Add count aggregation"""
        if property_name:
            self._aggregations[alias] = ("count", property_name)
        else:
            self._aggregations[alias] = "count"
        return self
    
    def sum(self, property_name: str, alias: str = None) -> 'QueryBuilder':
        """Add sum aggregation"""
        alias = alias or f"sum_{property_name}"
        self._aggregations[alias] = ("sum", property_name)
        return self
    
    def avg(self, property_name: str, alias: str = None) -> 'QueryBuilder':
        """Add average aggregation"""
        alias = alias or f"avg_{property_name}"
        self._aggregations[alias] = ("avg", property_name)
        return self
    
    def min(self, property_name: str, alias: str = None) -> 'QueryBuilder':
        """Add minimum aggregation"""
        alias = alias or f"min_{property_name}"
        self._aggregations[alias] = ("min", property_name)
        return self
    
    def max(self, property_name: str, alias: str = None) -> 'QueryBuilder':
        """Add maximum aggregation"""
        alias = alias or f"max_{property_name}"
        self._aggregations[alias] = ("max", property_name)
        return self
    
    def median(self, property_name: str, alias: str = None) -> 'QueryBuilder':
        """Add median aggregation"""
        alias = alias or f"median_{property_name}"
        self._aggregations[alias] = ("median", property_name)
        return self
    
    def percentile(self, property_name: str, percentile: float, alias: str = None) -> 'QueryBuilder':
        """Add percentile aggregation"""
        alias = alias or f"p{int(percentile*100)}_{property_name}"
        self._aggregations[alias] = ("percentile", {"property": property_name, "percentile": percentile})
        return self
    
    def collect(self, property_name: str, alias: str = None, unique: bool = False) -> 'QueryBuilder':
        """Collect property values"""
        alias = alias or f"collect_{property_name}"
        func = "collect_set" if unique else "collect"
        self._aggregations[alias] = (func, property_name)
        return self
    
    def having(self, condition: Callable) -> 'QueryBuilder':
        """Filter groups by condition"""
        self._having = condition
        return self
    
    def order_by(self, field: str, ascending: bool = False) -> 'QueryBuilder':
        """Order results"""
        self._order_by = field
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """Limit results"""
        self._limit = count
        return self
    
    def execute(self) -> AggregateResult:
        """Execute the query"""
        if not self._traversal:
            raise ValueError("No traversal specified. Use match() first.")
        
        # Get items from traversal
        items = self._traversal.to_list()
        
        # Apply aggregations
        return self.aggregation_engine.aggregate(
            items=items,
            group_by=self._group_by,
            aggregations=self._aggregations,
            having=self._having,
            order_by=self._order_by,
            limit=self._limit
        )


# Add aggregation methods to GraphTraversal class
class AggregatedTraversal(GraphTraversal):
    """Extended traversal with aggregation capabilities"""
    
    def __init__(self, storage: RedisStorage):
        super().__init__(storage)
        self.aggregation_engine = AggregationEngine()
    
    def group_by(self, *properties: str) -> QueryBuilder:
        """Start building an aggregation query"""
        builder = QueryBuilder(self.storage)
        return builder.match(self).group_by(*properties)
    
    def group_by_func(self, func: Callable) -> QueryBuilder:
        """Group by custom function"""
        builder = QueryBuilder(self.storage)
        return builder.match(self).group_by_func(func)
    
    def count_by(self, property_name: str) -> Dict[Any, int]:
        """Quick count by property"""
        items = self.to_list()
        result = self.aggregation_engine.aggregate(
            items, 
            group_by=property_name,
            aggregations={"count": "count"}
        )
        return {k: v["count"] for k, v in result.groups.items()}
    
    def sum_by(self, group_property: str, sum_property: str) -> Dict[Any, float]:
        """Quick sum by property"""
        items = self.to_list()
        result = self.aggregation_engine.aggregate(
            items,
            group_by=group_property,
            aggregations={"sum": ("sum", sum_property)}
        )
        return {k: v["sum"] for k, v in result.groups.items()}
    
    def avg_by(self, group_property: str, avg_property: str) -> Dict[Any, float]:
        """Quick average by property"""
        items = self.to_list()
        result = self.aggregation_engine.aggregate(
            items,
            group_by=group_property,
            aggregations={"avg": ("avg", avg_property)}
        )
        return {k: v["avg"] for k, v in result.groups.items()}
    
    # Statistical methods for entire result set
    def count(self) -> int:
        """Count all items"""
        return len(self.to_list())
    
    def sum(self, property_name: str) -> float:
        """Sum property across all items"""
        items = self.to_list()
        return self.aggregation_engine._sum(items, property_name)
    
    def avg(self, property_name: str) -> float:
        """Average property across all items"""
        items = self.to_list()
        return self.aggregation_engine._avg(items, property_name)
    
    def min(self, property_name: str) -> Any:
        """Minimum property value"""
        items = self.to_list()
        return self.aggregation_engine._min(items, property_name)
    
    def max(self, property_name: str) -> Any:
        """Maximum property value"""
        items = self.to_list()
        return self.aggregation_engine._max(items, property_name)
    
    def collect_values(self, property_name: str, unique: bool = False) -> List[Any]:
        """Collect all property values"""
        items = self.to_list()
        if unique:
            return self.aggregation_engine._collect_set(items, property_name)
        return self.aggregation_engine._collect(items, property_name)


# Update the main Graph class to use AggregatedTraversal
# This would go in mylath/graph/graph.py

def traversal(self) -> AggregatedTraversal:
    """Start a new graph traversal with aggregation support"""
    return AggregatedTraversal(self.storage)

def V(self, *node_ids) -> AggregatedTraversal:
    """Start traversal from vertices - shortcut method"""
    return self.traversal().V(list(node_ids) if node_ids else None)
