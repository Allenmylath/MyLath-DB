# DEBUGGING: Let's trace why projections aren't working

# Step 1: Check if projections are being extracted from physical plan
# Add this debug method to redis_executor.py

def _execute_project_fixed(self, operation, context) -> List[Dict[str, Any]]:
    """FIXED: Execute Project operation by executing children first - WITH PROJECTION DEBUG"""
    
    logger.debug("=== PROJECT OPERATION DEBUG ===")
    logger.debug(f"Operation type: {getattr(operation, 'operation_type', 'unknown')}")
    logger.debug(f"Has logical_op: {hasattr(operation, 'logical_op')}")
    
    if hasattr(operation, 'logical_op') and operation.logical_op:
        logical_op = operation.logical_op
        logger.debug(f"Logical op type: {type(logical_op).__name__}")
        logger.debug(f"Has projections: {hasattr(logical_op, 'projections')}")
        
        if hasattr(logical_op, 'projections'):
            logger.debug(f"Projections: {logical_op.projections}")
        
        # ALSO CHECK FOR RETURN ITEMS
        if hasattr(logical_op, 'items'):
            logger.debug(f"Return items: {logical_op.items}")
    
    # STEP 1: Execute child operations first to get base data
    base_results = []
    
    for child in operation.children:
        logger.debug(f"Executing Project child: {type(child).__name__} - {getattr(child, 'operation_type', 'unknown')}")
        child_results = self._execute_child_operation(child, context)
        base_results.extend(child_results)
        logger.debug(f"Child returned {len(child_results)} results")
    
    # STEP 2: If no base results, try to get them from logical operation
    if not base_results and hasattr(operation, 'logical_op'):
        logical_op = operation.logical_op
        if logical_op and hasattr(logical_op, 'children'):
            for child_logical in logical_op.children:
                logger.debug(f"Executing Project logical child: {type(child_logical).__name__}")
                child_results = self._execute_logical_operation(child_logical, context)
                base_results.extend(child_results)
                logger.debug(f"Logical child returned {len(child_results)} results")
    
    logger.debug(f"Base results before projection: {len(base_results)}")
    if base_results:
        logger.debug(f"Sample base result: {base_results[0]}")
    
    # STEP 3: FIXED - Apply projections using RETURN ITEMS instead of projections
    if base_results and hasattr(operation, 'logical_op'):
        logical_op = operation.logical_op
        
        # CHECK FOR RETURN ITEMS (this is likely where the projection info is)
        if hasattr(logical_op, 'items') and logical_op.items:
            logger.debug(f"Found {len(logical_op.items)} return items")
            projected_results = self._apply_return_items_to_results(base_results, logical_op.items)
            logger.debug(f"Applied return items: {len(projected_results)} results")
            if projected_results:
                logger.debug(f"Sample projected result: {projected_results[0]}")
            return projected_results
        
        # FALLBACK - Check for projections attribute
        elif hasattr(logical_op, 'projections') and logical_op.projections:
            logger.debug(f"Found {len(logical_op.projections)} projections")
            projected_results = self._apply_projections_to_results(base_results, logical_op.projections)
            logger.debug(f"Applied projections: {len(projected_results)} results")
            return projected_results
    
    # STEP 4: Return base results if no projections
    logger.debug(f"Project returning {len(base_results)} base results (no projections applied)")
    return base_results

# NEW METHOD: Handle return items (this is what we're missing!)
def _apply_return_items_to_results(self, results, return_items):
    """Apply return items to result set (this handles RETURN n.name, etc.)"""
    
    logger.debug(f"Applying {len(return_items)} return items to {len(results)} results")
    
    projected_results = []
    
    for result in results:
        projected_record = {}
        
        for return_item in return_items:
            try:
                # Extract expression and alias from return item
                expr = return_item.expression
                alias = return_item.alias
                
                logger.debug(f"Processing return item: {expr} (alias: {alias})")
                
                # Evaluate the expression
                value = self._evaluate_projection_expression(expr, result)
                
                # Use alias if provided, otherwise derive name from expression
                key = alias if alias else self._derive_expression_name(expr)
                projected_record[key] = value
                
                logger.debug(f"Added projection: {key} = {value}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate return item {return_item}: {e}")
                # Use null value for failed projections
                key = alias if alias else str(return_item)
                projected_record[key] = None
        
        projected_results.append(projected_record)
    
    return projected_results

# IMPROVED: Better property extraction
def _evaluate_projection_expression(self, expr, result):
    """IMPROVED: Evaluate projection expression with better property access"""
    
    from ..cypher_planner.ast_nodes import (
        PropertyExpression, VariableExpression, LiteralExpression,
        BinaryExpression, FunctionCall
    )
    
    logger.debug(f"Evaluating expression: {expr} (type: {type(expr).__name__})")
    
    if isinstance(expr, PropertyExpression):
        # Property access: variable.property -> extract specific property value
        logger.debug(f"Property expression: {expr.variable}.{expr.property_name}")
        
        entity = result.get(expr.variable)
        logger.debug(f"Entity for {expr.variable}: {entity}")
        
        if entity and isinstance(entity, dict):
            # Try different property access patterns
            
            # Pattern 1: Direct property access (e.g., entity['name'])
            if expr.property_name in entity:
                value = entity[expr.property_name]
                logger.debug(f"Found property directly: {expr.property_name} = {value}")
                return value
            
            # Pattern 2: Properties stored in 'properties' sub-dict
            elif 'properties' in entity and isinstance(entity['properties'], dict):
                value = entity['properties'].get(expr.property_name)
                logger.debug(f"Found property in properties dict: {expr.property_name} = {value}")
                return value
            
            # Pattern 3: Properties stored with _ prefix
            elif f'_{expr.property_name}' in entity:
                value = entity[f'_{expr.property_name}']
                logger.debug(f"Found property with underscore: _{expr.property_name} = {value}")
                return value
            
            # Pattern 4: Case-insensitive search
            for key, value in entity.items():
                if key.lower() == expr.property_name.lower():
                    logger.debug(f"Found property case-insensitive: {key} = {value}")
                    return value
        
        logger.debug(f"Property {expr.property_name} not found in entity")
        return None
        
    elif isinstance(expr, VariableExpression):
        # Simple variable reference
        logger.debug(f"Variable expression: {expr.name}")
        entity = result.get(expr.name)
        logger.debug(f"Variable value: {entity}")
        return entity
        
    elif isinstance(expr, LiteralExpression):
        # Literal value
        logger.debug(f"Literal expression: {expr.value}")
        return expr.value
        
    else:
        # Unknown expression type
        logger.debug(f"Unknown expression type: {type(expr)}")
        return str(expr)

# TESTING FUNCTION
def debug_projection_issue():
    """Debug why projections aren't working"""
    print("🔍 Debugging Projection Issue...")
    
    try:
        from mylathdb import MyLathDB
        from mylathdb.cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
        
        # Setup
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load data
        db.load_graph_data(nodes=[{'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']}])
        
        # Parse query manually to check AST
        query = "MATCH (n:Person) RETURN n.name"
        print(f"Parsing query: {query}")
        
        ast = parse_cypher_query(query)
        print(f"AST created: {type(ast).__name__}")
        
        if ast.return_clause:
            print(f"Return clause items: {len(ast.return_clause.items)}")
            for i, item in enumerate(ast.return_clause.items):
                print(f"  Item {i}: {item.expression} (alias: {item.alias})")
                print(f"    Expression type: {type(item.expression).__name__}")
                if hasattr(item.expression, 'variable'):
                    print(f"    Variable: {item.expression.variable}")
                if hasattr(item.expression, 'property_name'):
                    print(f"    Property: {item.expression.property_name}")
        
        # Create logical plan
        logical_planner = LogicalPlanner()
        logical_plan = logical_planner.create_logical_plan(ast)
        print(f"Logical plan: {type(logical_plan).__name__}")
        
        if hasattr(logical_plan, 'projections'):
            print(f"Logical plan projections: {logical_plan.projections}")
        if hasattr(logical_plan, 'items'):
            print(f"Logical plan items: {logical_plan.items}")
        
        # Create physical plan
        physical_planner = PhysicalPlanner()
        physical_plan = physical_planner.create_physical_plan(logical_plan)
        print(f"Physical plan: {type(physical_plan).__name__}")
        
        if hasattr(physical_plan, 'logical_op') and physical_plan.logical_op:
            log_op = physical_plan.logical_op
            print(f"Physical plan logical op: {type(log_op).__name__}")
            if hasattr(log_op, 'projections'):
                print(f"Physical logical op projections: {log_op.projections}")
            if hasattr(log_op, 'items'):
                print(f"Physical logical op items: {log_op.items}")
        
        print("✅ Debug info collected")
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# RUN THIS TO DEBUG:
if __name__ == "__main__":
    debug_projection_issue()