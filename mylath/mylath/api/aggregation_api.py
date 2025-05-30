# mylath/api/aggregation_api.py
from flask import request, jsonify
from typing import Dict, Any, List
from ..graph.aggregation import AggregationEngine, AggregateFunction
from ..graph.query_builder import QueryBuilder
from ..graph.traversal import GraphTraversal


class AggregationAPI:
    """REST API endpoints for aggregation operations"""
    
    def __init__(self, graph_api):
        self.graph = graph_api.graph
        self.app = graph_api.app
        self.aggregation_engine = AggregationEngine()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup aggregation API routes"""
        
        @self.app.route('/aggregations/simple', methods=['POST'])
        def simple_aggregation():
            """
            Simple aggregation endpoint
            
            Example payload:
            {
                "traversal": {
                    "steps": [
                        {"type": "V", "params": {}},
                        {"type": "has", "params": {"key": "label", "value": "person"}}
                    ]
                },
                "operation": "count"  // or "sum", "avg", etc.
                "property": "salary"  // required for sum, avg, etc.
            }
            """
            data = request.get_json()
            
            try:
                # Build traversal from steps
                traversal = self._build_traversal_from_steps(data.get('traversal', {}))
                items = traversal.to_list()
                
                operation = data.get('operation')
                property_name = data.get('property')
                
                if operation == 'count':
                    result = len(items)
                elif operation == 'sum':
                    result = self.aggregation_engine._sum(items, property_name)
                elif operation == 'avg':
                    result = self.aggregation_engine._avg(items, property_name)
                elif operation == 'min':
                    result = self.aggregation_engine._min(items, property_name)
                elif operation == 'max':
                    result = self.aggregation_engine._max(items, property_name)
                else:
                    return jsonify({"error": f"Unknown operation: {operation}"}), 400
                
                return jsonify({
                    "result": result,
                    "operation": operation,
                    "property": property_name,
                    "item_count": len(items)
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/aggregations/group-by', methods=['POST'])
        def group_by_aggregation():
            """
            Group by aggregation endpoint
            
            Example payload:
            {
                "traversal": {
                    "steps": [
                        {"type": "V", "params": {}},
                        {"type": "has", "params": {"key": "label", "value": "person"}}
                    ]
                },
                "group_by": "department",  // or ["city", "department"] for multiple
                "aggregations": {
                    "count": {"function": "count"},
                    "avg_salary": {"function": "avg", "property": "salary"},
                    "total_salary": {"function": "sum", "property": "salary"}
                },
                "having": {  // optional
                    "field": "count",
                    "operator": ">",
                    "value": 5
                },
                "order_by": "avg_salary",  // optional
                "limit": 10  // optional
            }
            """
            data = request.get_json()
            
            try:
                # Build traversal
                traversal = self._build_traversal_from_steps(data.get('traversal', {}))
                items = traversal.to_list()
                
                # Parse group by
                group_by = data.get('group_by')
                if isinstance(group_by, list) and len(group_by) == 1:
                    group_by = group_by[0]
                
                # Parse aggregations
                aggregations = {}
                for alias, agg_spec in data.get('aggregations', {}).items():
                    func = agg_spec['function']
                    prop = agg_spec.get('property')
                    
                    if prop:
                        aggregations[alias] = (func, prop)
                    else:
                        aggregations[alias] = func
                
                # Parse having clause
                having_func = None
                if 'having' in data:
                    having = data['having']
                    field = having['field']
                    operator = having['operator']
                    value = having['value']
                    
                    if operator == '>':
                        having_func = lambda g: g.get(field, 0) > value
                    elif operator == '<':
                        having_func = lambda g: g.get(field, 0) < value
                    elif operator == '>=':
                        having_func = lambda g: g.get(field, 0) >= value
                    elif operator == '<=':
                        having_func = lambda g: g.get(field, 0) <= value
                    elif operator == '==':
                        having_func = lambda g: g.get(field) == value
                
                # Execute aggregation
                result = self.aggregation_engine.aggregate(
                    items=items,
                    group_by=group_by,
                    aggregations=aggregations,
                    having=having_func,
                    order_by=data.get('order_by'),
                    limit=data.get('limit')
                )
                
                return jsonify(result.to_dict())
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/aggregations/advanced', methods=['POST'])
        def advanced_aggregation():
            """
            Advanced aggregation with query builder syntax
            
            Example payload:
            {
                "traversal": {
                    "steps": [
                        {"type": "V", "params": {}},
                        {"type": "has", "params": {"key": "label", "value": "person"}}
                    ]
                },
                "query": {
                    "group_by": ["department"],
                    "aggregations": [
                        {"alias": "headcount", "function": "count"},
                        {"alias": "avg_salary", "function": "avg", "property": "salary"},
                        {"alias": "p95_salary", "function": "percentile", "property": "salary", "percentile": 0.95},
                        {"alias": "all_names", "function": "collect", "property": "name", "unique": false}
                    ],
                    "having": {
                        "function": "lambda g: g['avg_salary'] > 80000"
                    },
                    "order_by": "avg_salary",
                    "limit": 5
                }
            }
            """
            data = request.get_json()
            
            try:
                # Build traversal
                traversal = self._build_traversal_from_steps(data.get('traversal', {}))
                
                # Build query using QueryBuilder
                builder = QueryBuilder(self.graph.storage)
                builder = builder.match(traversal)
                
                query_spec = data.get('query', {})
                
                # Add group by
                if 'group_by' in query_spec:
                    group_by = query_spec['group_by']
                    if len(group_by) == 1:
                        builder = builder.group_by(group_by[0])
                    else:
                        builder = builder.group_by(*group_by)
                
                # Add aggregations
                for agg in query_spec.get('aggregations', []):
                    alias = agg['alias']
                    func = agg['function']
                    prop = agg.get('property')
                    
                    if func == 'count':
                        builder = builder.count(alias, prop)
                    elif func == 'sum':
                        builder = builder.sum(prop, alias)
                    elif func == 'avg':
                        builder = builder.avg(prop, alias)
                    elif func == 'min':
                        builder = builder.min(prop, alias)
                    elif func == 'max':
                        builder = builder.max(prop, alias)
                    elif func == 'median':
                        builder = builder.median(prop, alias)
                    elif func == 'percentile':
                        percentile = agg.get('percentile', 0.5)
                        builder = builder.percentile(prop, percentile, alias)
                    elif func == 'collect':
                        unique = agg.get('unique', False)
                        builder = builder.collect(prop, alias, unique)
                
                # Add having clause (simplified - in production, would need proper parsing)
                if 'having' in query_spec:
                    # This is dangerous in production - would need proper sandboxing
                    having_code = query_spec['having']['function']
                    if having_code.startswith('lambda g:'):
                        having_func = eval(having_code)
                        builder = builder.having(having_func)
                
                # Add ordering and limit
                if 'order_by' in query_spec:
                    builder = builder.order_by(query_spec['order_by'])
                
                if 'limit' in query_spec:
                    builder = builder.limit(query_spec['limit'])
                
                # Execute
                result = builder.execute()
                return jsonify(result.to_dict())
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/aggregations/templates/<template_name>', methods=['POST'])
        def template_aggregation(template_name):
            """
            Predefined aggregation templates
            """
            data = request.get_json()
            
            try:
                if template_name == 'department_analysis':
                    return self._department_analysis_template(data)
                elif template_name == 'salary_stats':
                    return self._salary_stats_template(data)
                elif template_name == 'performance_distribution':
                    return self._performance_distribution_template(data)
                elif template_name == 'company_metrics':
                    return self._company_metrics_template(data)
                else:
                    return jsonify({"error": f"Unknown template: {template_name}"}), 400
                    
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/aggregations/pivot', methods=['POST'])
        def pivot_table():
            """
            Create pivot table from graph data
            
            Example payload:
            {
                "traversal": {
                    "steps": [
                        {"type": "V", "params": {}},
                        {"type": "has", "params": {"key": "label", "value": "person"}}
                    ]
                },
                "rows": ["department"],
                "columns": ["city"],
                "values": {"function": "avg", "property": "salary"},
                "fill_value": 0
            }
            """
            data = request.get_json()
            
            try:
                # Build traversal
                traversal = self._build_traversal_from_steps(data.get('traversal', {}))
                items = traversal.to_list()
                
                rows = data.get('rows', [])
                columns = data.get('columns', [])
                values_spec = data.get('values', {})
                fill_value = data.get('fill_value', 0)
                
                # Create pivot table
                pivot_result = self._create_pivot_table(items, rows, columns, values_spec, fill_value)
                
                return jsonify(pivot_result)
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def _build_traversal_from_steps(self, traversal_data: Dict[str, Any]) -> GraphTraversal:
        """Build GraphTraversal from API steps"""
        traversal = GraphTraversal(self.graph.storage)
        
        for step in traversal_data.get('steps', []):
            step_type = step.get('type')
            params = step.get('params', {})
            
            if step_type == 'V':
                node_ids = params.get('node_ids')
                traversal = traversal.V(node_ids)
            elif step_type == 'has':
                key = params.get('key')
                value = params.get('value')
                traversal = traversal.has(key, value)
            elif step_type == 'out':
                label = params.get('label')
                traversal = traversal.out(label)
            elif step_type == 'in':
                label = params.get('label')
                traversal = traversal.in_(label)
            elif step_type == 'limit':
                count = params.get('count')
                traversal = traversal.limit(count)
            # Add more step types as needed
        
        return traversal
    
    def _department_analysis_template(self, data):
        """Template for department analysis"""
        filters = data.get('filters', {})
        
        # Build base traversal
        traversal = GraphTraversal(self.graph.storage).V().has("label", "person")
        
        # Apply filters
        if 'min_age' in filters:
            traversal = traversal.filter(lambda n: n.properties.get('age', 0) >= filters['min_age'])
        if 'cities' in filters:
            traversal = traversal.filter(lambda n: n.properties.get('city') in filters['cities'])
        
        # Execute aggregation
        result = (QueryBuilder(self.graph.storage)
                 .match(traversal)
                 .group_by("department")
                 .count("headcount")
                 .avg("salary", "avg_salary")
                 .avg("age", "avg_age")
                 .avg("performance_score", "avg_performance")
                 .sum("salary", "total_payroll")
                 .execute())
        
        return jsonify(result.to_dict())
    
    def _salary_stats_template(self, data):
        """Template for salary statistics"""
        group_by = data.get('group_by', 'department')
        
        traversal = GraphTraversal(self.graph.storage).V().has("label", "person")
        
        result = (QueryBuilder(self.graph.storage)
                 .match(traversal)
                 .group_by(group_by)
                 .count("count")
                 .avg("salary", "avg_salary")
                 .median("salary", "median_salary")
                 .percentile("salary", 0.25, "p25_salary")
                 .percentile("salary", 0.75, "p75_salary")
                 .percentile("salary", 0.90, "p90_salary")
                 .execute())
        
        return jsonify(result.to_dict())
    
    def _performance_distribution_template(self, data):
        """Template for performance distribution analysis"""
        traversal = GraphTraversal(self.graph.storage).V().has("label", "person")
        
        # Custom grouping function for performance bands
        def performance_band_grouper(items, group_by_func):
            groups = {}
            for item in items:
                score = item.properties.get("performance_score", 0)
                if score >= 4.5:
                    band = "Excellent (4.5+)"
                elif score >= 4.0:
                    band = "Good (4.0-4.5)"
                elif score >= 3.0:
                    band = "Average (3.0-4.0)"
                else:
                    band = "Needs Improvement (<3.0)"
                
                if band not in groups:
                    groups[band] = []
                groups[band].append(item)
            return groups
        
        items = traversal.to_list()
        groups = performance_band_grouper(items, None)
        
        result_groups = {}
        for band, group_items in groups.items():
            result_groups[band] = {
                "count": len(group_items),
                "avg_salary": self.aggregation_engine._avg(group_items, "salary"),
                "avg_age": self.aggregation_engine._avg(group_items, "age")
            }
        
        return jsonify({
            "groups": result_groups,
            "total_groups": len(result_groups),
            "total_items": len(items)
        })
    
    def _company_metrics_template(self, data):
        """Template for company metrics"""
        results = {}
        
        for company in self.graph.V().has("label", "company").to_list():
            employees = self.graph.V(company.id).in_("works_at").to_list()
            
            if employees:
                company_metrics = {
                    "company_name": company.properties.get("name"),
                    "revenue": company.properties.get("revenue", 0),
                    "industry": company.properties.get("industry"),
                    "employee_count": len(employees),
                    "avg_salary": self.aggregation_engine._avg(employees, "salary"),
                    "total_payroll": self.aggregation_engine._sum(employees, "salary"),
                    "avg_performance": self.aggregation_engine._avg(employees, "performance_score"),
                    "avg_age": self.aggregation_engine._avg(employees, "age")
                }
                
                # Calculate derived metrics
                if company_metrics["revenue"] > 0:
                    company_metrics["revenue_per_employee"] = company_metrics["revenue"] / len(employees)
                    company_metrics["profit_margin_estimate"] = ((company_metrics["revenue"] - company_metrics["total_payroll"]) / company_metrics["revenue"]) * 100
                
                results[company.id] = company_metrics
        
        return jsonify({"companies": results})
    
    def _create_pivot_table(self, items, rows, columns, values_spec, fill_value):
        """Create a pivot table from items"""
        # Get unique values for rows and columns
        row_values = set()
        col_values = set()
        
        for item in items:
            row_key = tuple(item.properties.get(r) for r in rows) if len(rows) > 1 else item.properties.get(rows[0])
            col_key = tuple(item.properties.get(c) for c in columns) if len(columns) > 1 else item.properties.get(columns[0])
            
            row_values.add(row_key)
            col_values.add(col_key)
        
        row_values = sorted(list(row_values))
        col_values = sorted(list(col_values))
        
        # Initialize pivot table
        pivot_data = {}
        for row_val in row_values:
            pivot_data[row_val] = {}
            for col_val in col_values:
                pivot_data[row_val][col_val] = []
        
        # Populate pivot table
        for item in items:
            row_key = tuple(item.properties.get(r) for r in rows) if len(rows) > 1 else item.properties.get(rows[0])
            col_key = tuple(item.properties.get(c) for c in columns) if len(columns) > 1 else item.properties.get(columns[0])
            
            if row_key in pivot_data and col_key in pivot_data[row_key]:
                pivot_data[row_key][col_key].append(item)
        
        # Apply aggregation function
        func = values_spec.get('function', 'count')
        prop = values_spec.get('property')
        
        result_table = {}
        for row_val in row_values:
            result_table[row_val] = {}
            for col_val in col_values:
                cell_items = pivot_data[row_val][col_val]
                
                if func == 'count':
                    value = len(cell_items)
                elif func == 'sum':
                    value = self.aggregation_engine._sum(cell_items, prop) if cell_items else fill_value
                elif func == 'avg':
                    value = self.aggregation_engine._avg(cell_items, prop) if cell_items else fill_value
                elif func == 'min':
                    value = self.aggregation_engine._min(cell_items, prop) if cell_items else fill_value
                elif func == 'max':
                    value = self.aggregation_engine._max(cell_items, prop) if cell_items else fill_value
                else:
                    value = len(cell_items)
                
                result_table[row_val][col_val] = value
        
        return {
            "pivot_table": result_table,
            "rows": row_values,
            "columns": col_values,
            "aggregation": values_spec
        }


