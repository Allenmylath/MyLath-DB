# mylath/api/enhanced_graph_api.py
"""
Enhanced GraphAPI with aggregation support
"""

from flask import Flask, request, jsonify
from typing import Dict, Any
from ..graph.graph import Graph
from ..storage.redis_storage import RedisStorage
from .aggregation_api import AggregationAPI
import json


class EnhancedGraphAPI:
    """Enhanced REST API for MyLath with aggregation support"""
    
    def __init__(self, storage: RedisStorage):
        self.app = Flask(__name__)
        self.graph = Graph(storage)
        
        # Initialize aggregation API
        self.aggregation_api = AggregationAPI(self)
        
        # Setup core routes (node, edge, vector operations)
        self._setup_core_routes()
        
        # Setup enhanced query routes
        self._setup_enhanced_routes()
    
    def _setup_core_routes(self):
        """Setup basic CRUD routes (same as original GraphAPI)"""
        
        @self.app.route('/nodes', methods=['POST'])
        def create_node():
            data = request.get_json()
            label = data.get('label')
            properties = data.get('properties', {})
            
            if not label:
                return jsonify({"error": "Label is required"}), 400
                
            node = self.graph.create_node(label, properties)
            return jsonify({
                "id": node.id,
                "label": node.label,
                "properties": node.properties,
                "created_at": node.created_at
            }), 201
        
        @self.app.route('/nodes/<node_id>', methods=['GET'])
        def get_node(node_id):
            node = self.graph.get_node(node_id)
            if not node:
                return jsonify({"error": "Node not found"}), 404
                
            return jsonify({
                "id": node.id,
                "label": node.label,
                "properties": node.properties,
                "created_at": node.created_at
            })
        
        # Add other CRUD operations...
        
    def _setup_enhanced_routes(self):
        """Setup enhanced query routes with aggregation"""
        
        @self.app.route('/query/analytics', methods=['POST'])
        def analytics_query():
            """
            Advanced analytics endpoint combining traversal + aggregation
            
            Example payload:
            {
                "name": "Department Salary Analysis",
                "description": "Analyze salary distribution by department",
                "base_query": {
                    "type": "traversal",
                    "steps": [
                        {"type": "V", "params": {}},
                        {"type": "has", "params": {"key": "label", "value": "person"}},
                        {"type": "filter", "params": {"condition": "age >= 25"}}
                    ]
                },
                "analytics": {
                    "type": "aggregation",
                    "group_by": ["department", "city"],
                    "metrics": [
                        {"name": "headcount", "function": "count"},
                        {"name": "avg_salary", "function": "avg", "property": "salary"},
                        {"name": "salary_range", "function": "custom", "expression": "max(salary) - min(salary)"},
                        {"name": "top_performers", "function": "filter_count", "condition": "performance_score >= 4.0"}
                    ],
                    "filters": {
                        "having": "headcount >= 3",
                        "order_by": "avg_salary DESC",
                        "limit": 10
                    }
                },
                "format": "summary"  // or "detailed", "pivot", "chart_data"
            }
            """
            data = request.get_json()
            
            try:
                # Build base traversal
                base_query = data.get('base_query', {})
                traversal = self._build_traversal_from_query(base_query)
                items = traversal.to_list()
                
                # Apply analytics
                analytics = data.get('analytics', {})
                result = self._apply_analytics(items, analytics)
                
                # Format result
                format_type = data.get('format', 'detailed')
                formatted_result = self._format_analytics_result(result, format_type)
                
                return jsonify({
                    "query_name": data.get('name', 'Unnamed Query'),
                    "description": data.get('description', ''),
                    "total_items": len(items),
                    "result": formatted_result,
                    "execution_time_ms": 0  # Would track actual execution time
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/query/insights', methods=['POST'])
        def insights_query():
            """
            Generate business insights from graph data
            
            Example payload:
            {
                "domain": "hr_analytics",  // or "sales", "marketing", etc.
                "focus": "retention_risk",
                "parameters": {
                    "time_period": "last_12_months",
                    "departments": ["Engineering", "Sales"],
                    "risk_factors": ["low_performance", "high_salary_vs_market", "low_engagement"]
                }
            }
            """
            data = request.get_json()
            
            try:
                domain = data.get('domain')
                focus = data.get('focus')
                params = data.get('parameters', {})
                
                if domain == 'hr_analytics':
                    insights = self._generate_hr_insights(focus, params)
                elif domain == 'network_analysis':
                    insights = self._generate_network_insights(focus, params)
                elif domain == 'project_analytics':
                    insights = self._generate_project_insights(focus, params)
                else:
                    return jsonify({"error": f"Unknown domain: {domain}"}), 400
                
                return jsonify(insights)
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/query/cohort-analysis', methods=['POST'])
        def cohort_analysis():
            """
            Cohort analysis endpoint
            
            Example payload:
            {
                "cohort_definition": {
                    "type": "hire_date",
                    "property": "start_date",
                    "grouping": "monthly"  // or "quarterly", "yearly"
                },
                "metrics": [
                    {"name": "retention_rate", "definition": "still_employed"},
                    {"name": "avg_performance", "property": "performance_score"},
                    {"name": "promotion_rate", "definition": "role_changes"}
                ],
                "time_periods": 12  // months to track
            }
            """
            data = request.get_json()
            
            try:
                cohort_def = data.get('cohort_definition', {})
                metrics = data.get('metrics', [])
                time_periods = data.get('time_periods', 12)
                
                cohort_result = self._perform_cohort_analysis(cohort_def, metrics, time_periods)
                
                return jsonify(cohort_result)
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/query/correlation', methods=['POST'])
        def correlation_analysis():
            """
            Correlation analysis between properties
            
            Example payload:
            {
                "base_query": {
                    "type": "traversal",
                    "steps": [
                        {"type": "V", "params": {}},
                        {"type": "has", "params": {"key": "label", "value": "person"}}
                    ]
                },
                "variables": [
                    {"name": "salary", "property": "salary"},
                    {"name": "performance", "property": "performance_score"},
                    {"name": "experience", "property": "experience_years"},
                    {"name": "network_size", "calculation": "out_degree('knows')"}
                ],
                "method": "pearson"  // or "spearman", "kendall"
            }
            """
            data = request.get_json()
            
            try:
                # Build base traversal
                base_query = data.get('base_query', {})
                traversal = self._build_traversal_from_query(base_query)
                items = traversal.to_list()
                
                # Extract variables
                variables = data.get('variables', [])
                method = data.get('method', 'pearson')
                
                correlation_result = self._calculate_correlations(items, variables, method)
                
                return jsonify(correlation_result)
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def _build_traversal_from_query(self, query_data):
        """Build traversal from query specification"""
        from ..graph.traversal import GraphTraversal
        
        traversal = GraphTraversal(self.graph.storage)
        
        if query_data.get('type') == 'traversal':
            for step in query_data.get('steps', []):
                step_type = step.get('type')
                params = step.get('params', {})
                
                if step_type == 'V':
                    node_ids = params.get('node_ids')
                    traversal = traversal.V(node_ids)
                elif step_type == 'has':
                    key = params.get('key')
                    value = params.get('value')
                    traversal = traversal.has(key, value)
                elif step_type == 'filter':
                    condition = params.get('condition')
                    # Simple condition parsing (in production, would need proper parser)
                    if 'age >= 25' in condition:
                        traversal = traversal.filter(lambda n: n.properties.get('age', 0) >= 25)
                # Add more step types as needed
        
        return traversal
    
    def _apply_analytics(self, items, analytics_spec):
        """Apply analytics specification to items"""
        from ..graph.query_builder import QueryBuilder
        from ..graph.traversal import GraphTraversal
        
        # Create a dummy traversal with the items
        # In practice, would need a better way to handle this
        analytics_type = analytics_spec.get('type')
        
        if analytics_type == 'aggregation':
            # Use aggregation engine directly
            group_by = analytics_spec.get('group_by')
            metrics = analytics_spec.get('metrics', [])
            
            # Convert metrics to aggregations dict
            aggregations = {}
            for metric in metrics:
                name = metric['name']
                function = metric['function']
                property_name = metric.get('property')
                
                if function == 'custom':
                    # Handle custom expressions (simplified)
                    expression = metric.get('expression', '')
                    if 'max(salary) - min(salary)' in expression:
                        # Calculate range manually
                        max_val = self.aggregation_api.aggregation_engine._max(items, 'salary')
                        min_val = self.aggregation_api.aggregation_engine._min(items, 'salary')
                        # Store as a post-processing step
                        aggregations[name] = 'count'  # Placeholder
                elif function == 'filter_count':
                    condition = metric.get('condition', '')
                    if 'performance_score >= 4.0' in condition:
                        filtered_items = [item for item in items if item.properties.get('performance_score', 0) >= 4.0]
                        # Store result directly
                        aggregations[name] = len(filtered_items)
                else:
                    if property_name:
                        aggregations[name] = (function, property_name)
                    else:
                        aggregations[name] = function
            
            # Apply filters
            filters = analytics_spec.get('filters', {})
            having = None
            if 'having' in filters:
                having_expr = filters['having']
                if 'headcount >= 3' in having_expr:
                    having = lambda g: g.get('headcount', 0) >= 3
            
            result = self.aggregation_api.aggregation_engine.aggregate(
                items=items,
                group_by=group_by,
                aggregations=aggregations,
                having=having,
                order_by=filters.get('order_by'),
                limit=filters.get('limit')
            )
            
            return result
        
        return None
    
    def _format_analytics_result(self, result, format_type):
        """Format analytics result based on format type"""
        if format_type == 'summary':
            return {
                "summary": {
                    "total_groups": result.total_groups,
                    "total_items": result.total_items,
                    "top_groups": dict(list(result.groups.items())[:5])
                }
            }
        elif format_type == 'pivot':
            # Convert to pivot table format
            return self._convert_to_pivot(result)
        elif format_type == 'chart_data':
            # Convert to chart-friendly format
            return self._convert_to_chart_data(result)
        else:
            return result.to_dict()
    
    def _generate_hr_insights(self, focus, params):
        """Generate HR analytics insights"""
        insights = {"domain": "hr_analytics", "focus": focus, "insights": []}
        
        if focus == 'retention_risk':
            # Analyze retention risk factors
            high_risk_employees = []
            
            for person in self.graph.V().has("label", "person").to_list():
                risk_score = 0
                risk_factors = []
                
                # Low performance
                performance = person.properties.get('performance_score', 3.0)
                if performance < 3.0:
                    risk_score += 30
                    risk_factors.append('low_performance')
                
                # High salary (assumption: top 10% might be flight risk)
                salary = person.properties.get('salary', 0)
                avg_dept_salary = self._get_avg_dept_salary(person.properties.get('department'))
                if salary > avg_dept_salary * 1.3:
                    risk_score += 20
                    risk_factors.append('high_salary')
                
                # Low network connectivity
                friend_count = self.graph.V(person.id).out("knows").count()
                if friend_count < 2:
                    risk_score += 25
                    risk_factors.append('low_connectivity')
                
                if risk_score >= 50:
                    high_risk_employees.append({
                        "employee": person.properties.get('name'),
                        "department": person.properties.get('department'),
                        "risk_score": risk_score,
                        "risk_factors": risk_factors
                    })
            
            insights['insights'].append({
                "type": "high_risk_employees",
                "count": len(high_risk_employees),
                "employees": high_risk_employees[:10],  # Top 10
                "recommendation": "Consider retention strategies for high-risk employees"
            })
        
        return insights
    
    def _get_avg_dept_salary(self, department):
        """Get average salary for department"""
        if not department:
            return 80000  # Default
        
        dept_employees = self.graph.V().has("department", department).to_list()
        if not dept_employees:
            return 80000
        
        return self.aggregation_api.aggregation_engine._avg(dept_employees, "salary")
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the enhanced API server"""
        self.app.run(host=host, port=port, debug=debug)
