#!/usr/bin/env python3
"""
Comprehensive Test Suite for Cypher Parser
Measures performance, detects issues, and validates functionality
"""

import time
import sys
import gc
import tracemalloc
import psutil
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

# Add current directory to path for imports
sys.path.insert(0, '.')

# Import cypher_planner components
try:
    from cypher_planner import (
        CypherParser, QueryPlanner, 
        parse_cypher_query, validate_cypher_query, get_cypher_errors,
        CypherParserError, get_package_info
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import Error: {e}")
    print("Some tests may be skipped")
    IMPORTS_AVAILABLE = False

@dataclass
class TestResult:
    """Results from a single test"""
    test_name: str
    query: str
    success: bool
    parse_time: float  # milliseconds
    plan_time: float   # milliseconds
    memory_used: float # MB
    error_message: str = ""
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

@dataclass
class TestSuite:
    """Complete test suite results"""
    name: str
    results: List[TestResult]
    total_time: float
    success_rate: float
    avg_parse_time: float
    avg_plan_time: float
    peak_memory: float
    issues_found: List[str] = None
    
    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []

class CypherTestSuite:
    """Comprehensive test suite for Cypher parser"""
    
    def __init__(self):
        self.parser = None
        self.planner = None
        self.results = []
        self.process = psutil.Process(os.getpid())
        
        if IMPORTS_AVAILABLE:
            self.parser = CypherParser()
            self.planner = QueryPlanner()
    
    def run_all_tests(self) -> Dict[str, TestSuite]:
        """Run all test suites and return comprehensive results"""
        print("🚀 Starting Comprehensive Cypher Parser Test Suite")
        print("=" * 60)
        
        if not IMPORTS_AVAILABLE:
            print("❌ Cannot run tests - imports failed")
            return {}
        
        # Show system info
        self._print_system_info()
        
        test_suites = {
            'performance': self._run_performance_tests(),
            'error_handling': self._run_error_handling_tests(),
            'edge_cases': self._run_edge_case_tests(),
            'stress': self._run_stress_tests(),
            'regression': self._run_regression_tests(),
            'memory': self._run_memory_tests(),
            'scalability': self._run_scalability_tests()
        }
        
        # Generate comprehensive report
        self._generate_comprehensive_report(test_suites)
        
        return test_suites
    
    def _print_system_info(self):
        """Print system information"""
        print("🖥️ System Information:")
        print(f"   Python Version: {sys.version.split()[0]}")
        print(f"   Platform: {sys.platform}")
        print(f"   CPU Count: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        if IMPORTS_AVAILABLE:
            info = get_package_info()
            print(f"   Cypher Planner: {info['version']}")
        print()
    
    def _run_performance_tests(self) -> TestSuite:
        """Test parsing performance across different query types"""
        print("⚡ Running Performance Tests...")
        
        perf_queries = [
            # Basic queries (should be fast)
            ("Basic Node", "MATCH (n) RETURN n", "simple"),
            ("Basic Label", "MATCH (n:Person) RETURN n.name", "simple"),
            ("Basic Relationship", "MATCH (a)-[r]->(b) RETURN a, b", "simple"),
            
            # Medium complexity
            ("Property Filter", "MATCH (n:Person) WHERE n.age > 25 AND n.country = 'USA' RETURN n.name", "medium"),
            ("Relationship Types", "MATCH (a:Person)-[r:KNOWS|FOLLOWS]->(b:Person) RETURN a.name, b.name", "medium"),
            ("Simple Variable Length", "MATCH (a)-[*1..3]->(b) RETURN a, b", "medium"),
            
            # High complexity
            ("Complex Variable Length", "MATCH (a:Person)-[r:KNOWS*2..5]->(b:Person) WHERE a.age > b.age RETURN a.name, b.name", "complex"),
            ("Multiple Clauses", """
                MATCH (p:Person {country: 'USA'})
                WHERE p.age > 21
                WITH p, p.friends as friends
                WHERE size(friends) > 2
                RETURN p.name, size(friends)
            """, "complex"),
            ("Nested Expressions", """
                MATCH (a:Person)-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company)
                WHERE a.age > 25 AND b.salary > 50000 AND c.location = 'NYC'
                RETURN a.name, collect(distinct c.name) as companies
                ORDER BY size(companies) DESC
                LIMIT 10
            """, "complex"),
            
            # Very high complexity
            ("Deep Nesting", """
                MATCH (a:Person)-[:KNOWS*1..4]->(b:Person)-[:WORKS_AT]->(c:Company)
                WHERE a.age BETWEEN 25 AND 65 
                  AND b.salary > average_salary(c)
                  AND c.industry IN ['tech', 'finance']
                WITH a, collect(b) as colleagues, collect(c) as companies
                WHERE size(colleagues) > 5 AND size(companies) > 2
                UNWIND colleagues as colleague
                MATCH (colleague)-[:LIVES_IN]->(city:City)
                RETURN a.name as person,
                       size(colleagues) as colleague_count,
                       collect(distinct city.name) as cities
                ORDER BY colleague_count DESC, size(cities) DESC
                LIMIT 20
            """, "very_complex"),
        ]
        
        results = []
        categories = defaultdict(list)
        
        for test_name, query, category in perf_queries:
            result = self._measure_single_query(test_name, query)
            results.append(result)
            categories[category].append(result.parse_time)
            
            # Print immediate results
            status = "✅" if result.success else "❌"
            print(f"   {status} {test_name:<25} Parse: {result.parse_time:6.2f}ms  Plan: {result.plan_time:6.2f}ms")
        
        # Analyze performance by category
        print("\n📊 Performance by Category:")
        for category, times in categories.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"   {category:<12}: avg={avg_time:6.2f}ms, min={min_time:6.2f}ms, max={max_time:6.2f}ms")
        
        total_time = sum(r.parse_time + r.plan_time for r in results)
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        avg_parse = sum(r.parse_time for r in results) / len(results)
        avg_plan = sum(r.plan_time for r in results) / len(results)
        peak_memory = max(r.memory_used for r in results)
        
        return TestSuite("Performance", results, total_time, success_rate, avg_parse, avg_plan, peak_memory)
    
    def _run_error_handling_tests(self) -> TestSuite:
        """Test error handling and recovery capabilities"""
        print("\n🔍 Running Error Handling Tests...")
        
        error_queries = [
            # Syntax errors
            ("Missing Parenthesis", "MATCH (n:Person WHERE n.age > 25 RETURN n"),
            ("Unbalanced Brackets", "MATCH (n)-[r:KNOWS->(b) RETURN n, b"),
            ("Invalid Operator", "MATCH (n) WHERE n.age >> 25 RETURN n"),
            ("Incomplete Pattern", "MATCH (a)-[r]- RETURN a"),
            ("Empty WHERE", "MATCH (n) WHERE RETURN n"),
            ("Invalid Variable Length", "MATCH (a)-[*-1..3]->(b) RETURN a, b"),
            ("Unclosed String", "MATCH (n {name: 'John}) RETURN n"),
            
            # Semantic errors
            ("Undefined Variable", "MATCH (n:Person) RETURN m.name"),
            ("Mixed Aggregation", "MATCH (n) RETURN count(n) + n.name"),
            ("Invalid Function", "MATCH (n) RETURN nonexistent_func(n)"),
            ("Circular Reference", "MATCH (a)-[r]->(a) WHERE r.weight = r.weight + 1 RETURN a"),
            
            # Edge cases
            ("Very Long Query", "MATCH " + "(n)-[r]->" * 50 + "(m) RETURN n, m"),
            ("Deep Nesting", "MATCH (a" + "-[:REL]->(b" * 20 + ") RETURN a"),
            ("Many Variables", "MATCH " + ", ".join(f"(n{i}:Type{i})" for i in range(100)) + " RETURN " + ", ".join(f"n{i}" for i in range(100))),
            
            # Unicode and special characters
            ("Unicode Labels", "MATCH (n:Personë) WHERE n.nämé = 'Jöhn' RETURN n"),
            ("Special Characters", "MATCH (n {`weird-property`: 'value'}) RETURN n"),
            ("Emoji in Query", "MATCH (n:Person {mood: '😊'}) RETURN n"),
        ]
        
        results = []
        error_categories = defaultdict(int)
        
        for test_name, query in error_queries:
            result = self._measure_single_query(test_name, query, expect_error=True)
            results.append(result)
            
            # Categorize errors
            if not result.success and result.error_message:
                if "syntax" in result.error_message.lower():
                    error_categories["syntax"] += 1
                elif "semantic" in result.error_message.lower() or "undefined" in result.error_message.lower():
                    error_categories["semantic"] += 1
                elif "memory" in result.error_message.lower():
                    error_categories["memory"] += 1
                else:
                    error_categories["other"] += 1
            
            # Print immediate results
            status = "✅" if not result.success else "⚠️"  # For error tests, failure is success
            print(f"   {status} {test_name:<25} Parse: {result.parse_time:6.2f}ms")
        
        print("\n📊 Error Categories:")
        for category, count in error_categories.items():
            print(f"   {category:<12}: {count} errors")
        
        total_time = sum(r.parse_time + r.plan_time for r in results)
        # For error tests, success rate is the rate of proper error detection
        success_rate = sum(1 for r in results if not r.success) / len(results) * 100
        avg_parse = sum(r.parse_time for r in results) / len(results)
        avg_plan = sum(r.plan_time for r in results) / len(results)
        peak_memory = max(r.memory_used for r in results)
        
        return TestSuite("Error Handling", results, total_time, success_rate, avg_parse, avg_plan, peak_memory)
    
    def _run_edge_case_tests(self) -> TestSuite:
        """Test edge cases and boundary conditions"""
        print("\n🎯 Running Edge Case Tests...")
        
        edge_queries = [
            # Empty/minimal queries
            ("Empty Query", ""),
            ("Whitespace Only", "   \n\t  "),
            ("Comment Only", "// This is just a comment"),
            ("Just MATCH", "MATCH"),
            ("Just RETURN", "RETURN"),
            
            # Boundary values
            ("Zero Variable Length", "MATCH (a)-[*0..0]->(b) RETURN a, b"),
            ("Large Variable Length", "MATCH (a)-[*1..100]->(b) RETURN a, b"),
            ("Infinite Variable Length", "MATCH (a)-[*]->(b) RETURN a, b"),
            ("Single Character Names", "MATCH (a)-[r]->(b) RETURN a"),
            ("Very Long Names", f"MATCH (very_long_variable_name_{'x' * 100}:VeryLongLabelName) RETURN very_long_variable_name_{'x' * 100}"),
            
            # Complex patterns
            ("Bidirectional", "MATCH (a)--(b) RETURN a, b"),
            ("Self Reference", "MATCH (a)-[r]->(a) RETURN a"),
            ("Multiple Self Refs", "MATCH (a)-[r1]->(a)-[r2]->(a) RETURN a"),
            ("Star Pattern", "MATCH (center)<--(a), (center)<--(b), (center)<--(c) RETURN center, a, b, c"),
            
            # Data type edge cases
            ("Null Values", "MATCH (n) WHERE n.value = null RETURN n"),
            ("Boolean Values", "MATCH (n) WHERE n.active = true AND n.deleted = false RETURN n"),
            ("Large Numbers", "MATCH (n) WHERE n.value = 999999999999999 RETURN n"),
            ("Scientific Notation", "MATCH (n) WHERE n.value = 1.23e-10 RETURN n"),
            ("Negative Numbers", "MATCH (n) WHERE n.value = -999.999 RETURN n"),
            
            # Escaping and quoting
            ("Escaped Quotes", r"MATCH (n {name: 'John\'s House'}) RETURN n"),
            ("Double Quotes", 'MATCH (n {name: "John\'s House"}) RETURN n'),
            ("Mixed Quotes", """MATCH (n {name: "John's 'special' house"}) RETURN n"""),
            ("Backtick Names", "MATCH (n:`Weird Label`) RETURN n.`weird property`"),
        ]
        
        results = []
        for test_name, query in edge_queries:
            result = self._measure_single_query(test_name, query)
            results.append(result)
            
            # Print immediate results
            status = "✅" if result.success else "❌"
            print(f"   {status} {test_name:<25} Parse: {result.parse_time:6.2f}ms")
        
        total_time = sum(r.parse_time + r.plan_time for r in results)
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        avg_parse = sum(r.parse_time for r in results) / len(results)
        avg_plan = sum(r.plan_time for r in results) / len(results)
        peak_memory = max(r.memory_used for r in results)
        
        return TestSuite("Edge Cases", results, total_time, success_rate, avg_parse, avg_plan, peak_memory)
    
    def _run_stress_tests(self) -> TestSuite:
        """Stress test with high volume and complexity"""
        print("\n💪 Running Stress Tests...")
        
        results = []
        
        # Test 1: High volume of simple queries
        simple_query = "MATCH (n:Person) WHERE n.age > 25 RETURN n.name"
        print("   Testing high volume (1000 simple queries)...")
        
        start_time = time.time()
        successes = 0
        total_parse_time = 0
        
        for i in range(1000):
            result = self._measure_single_query(f"Volume_{i}", simple_query, silent=True)
            if result.success:
                successes += 1
            total_parse_time += result.parse_time
        
        volume_time = time.time() - start_time
        print(f"      Completed: {successes}/1000 successful, {total_parse_time:.1f}ms total parse time")
        
        results.append(TestResult("High Volume", "1000x simple queries", 
                                successes > 990, total_parse_time, 0, 0))
        
        # Test 2: Memory stress test
        print("   Testing memory usage with large queries...")
        large_query = "MATCH " + "".join(f"(n{i}:Type{i})-[r{i}:REL{i}]->" for i in range(50)) + "(final) RETURN " + ", ".join(f"n{i}" for i in range(50))
        
        tracemalloc.start()
        result = self._measure_single_query("Memory Stress", large_query)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results.append(result)
        print(f"      Memory usage: {peak / 1024 / 1024:.2f} MB peak")
        
        # Test 3: Deep recursion test
        print("   Testing deep query nesting...")
        deep_query = "MATCH " + "-[:REL]->(n)" * 100 + " RETURN n"
        result = self._measure_single_query("Deep Nesting", deep_query)
        results.append(result)
        
        total_time = sum(r.parse_time + r.plan_time for r in results)
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        avg_parse = sum(r.parse_time for r in results) / len(results)
        avg_plan = sum(r.plan_time for r in results) / len(results)
        peak_memory = max(r.memory_used for r in results)
        
        return TestSuite("Stress Tests", results, total_time, success_rate, avg_parse, avg_plan, peak_memory)
    
    def _run_regression_tests(self) -> TestSuite:
        """Test core functionality to ensure no regressions"""
        print("\n🔄 Running Regression Tests...")
        
        regression_queries = [
            # Core MATCH patterns
            ("Node Match", "MATCH (n) RETURN n"),
            ("Label Match", "MATCH (n:Person) RETURN n"),
            ("Property Match", "MATCH (n {name: 'John'}) RETURN n"),
            ("Relationship Match", "MATCH (a)-[r]->(b) RETURN a, r, b"),
            ("Typed Relationship", "MATCH (a)-[r:KNOWS]->(b) RETURN a, r, b"),
            
            # WHERE clauses
            ("Simple WHERE", "MATCH (n) WHERE n.age > 25 RETURN n"),
            ("Complex WHERE", "MATCH (n) WHERE n.age > 25 AND n.name = 'John' RETURN n"),
            ("OR Condition", "MATCH (n) WHERE n.age > 65 OR n.age < 18 RETURN n"),
            
            # RETURN variations
            ("Simple RETURN", "MATCH (n) RETURN n"),
            ("Property RETURN", "MATCH (n) RETURN n.name, n.age"),
            ("Alias RETURN", "MATCH (n) RETURN n.name AS name, n.age AS age"),
            ("DISTINCT", "MATCH (n) RETURN DISTINCT n.type"),
            
            # ORDER BY and LIMIT
            ("ORDER BY", "MATCH (n) RETURN n ORDER BY n.name"),
            ("ORDER BY DESC", "MATCH (n) RETURN n ORDER BY n.age DESC"),
            ("LIMIT", "MATCH (n) RETURN n LIMIT 10"),
            ("SKIP LIMIT", "MATCH (n) RETURN n SKIP 5 LIMIT 10"),
            
            # Variable length paths
            ("Variable Length", "MATCH (a)-[*1..3]->(b) RETURN a, b"),
            ("Unbounded Path", "MATCH (a)-[*]->(b) RETURN a, b"),
            
            # OPTIONAL MATCH
            ("Optional Match", "MATCH (a) OPTIONAL MATCH (a)-[r]->(b) RETURN a, b"),
            
            # WITH clauses
            ("WITH Clause", "MATCH (a) WITH a, a.name AS name WHERE name IS NOT NULL RETURN name"),
            
            # Functions
            ("Count Function", "MATCH (n) RETURN count(n)"),
            ("Collect Function", "MATCH (a)-[]->(b) RETURN a, collect(b)"),
        ]
        
        results = []
        for test_name, query in regression_queries:
            result = self._measure_single_query(test_name, query)
            results.append(result)
            
            # Print immediate results
            status = "✅" if result.success else "❌"
            print(f"   {status} {test_name:<20} Parse: {result.parse_time:6.2f}ms")
        
        total_time = sum(r.parse_time + r.plan_time for r in results)
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        avg_parse = sum(r.parse_time for r in results) / len(results)
        avg_plan = sum(r.plan_time for r in results) / len(results)
        peak_memory = max(r.memory_used for r in results)
        
        return TestSuite("Regression", results, total_time, success_rate, avg_parse, avg_plan, peak_memory)
    
    def _run_memory_tests(self) -> TestSuite:
        """Test memory usage and potential leaks"""
        print("\n🧠 Running Memory Tests...")
        
        results = []
        
        # Test memory growth with repeated parsing
        print("   Testing for memory leaks...")
        base_query = "MATCH (n:Person)-[r:KNOWS]->(m:Person) WHERE n.age > m.age RETURN n.name, m.name"
        
        memory_samples = []
        for i in range(100):
            gc.collect()  # Force garbage collection
            mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
            
            result = self._measure_single_query(f"Memory_{i}", base_query, silent=True)
            
            mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(mem_after - mem_before)
        
        avg_memory_delta = sum(memory_samples) / len(memory_samples)
        max_memory_delta = max(memory_samples)
        
        print(f"      Average memory delta: {avg_memory_delta:.3f} MB")
        print(f"      Maximum memory delta: {max_memory_delta:.3f} MB")
        
        # Check for significant memory growth
        memory_leak_detected = avg_memory_delta > 0.1  # More than 0.1MB average growth
        
        results.append(TestResult("Memory Leak Test", "100x repeated parsing", 
                                not memory_leak_detected, 0, 0, max_memory_delta,
                                "Potential memory leak detected" if memory_leak_detected else ""))
        
        # Test large query memory usage
        print("   Testing large query memory usage...")
        large_elements = []
        for i in range(1000):
            large_elements.append(f"(n{i}:Type{i} {{prop{i}: 'value{i}'}})")
        
        large_query = "MATCH " + ", ".join(large_elements) + " RETURN " + ", ".join(f"n{i}" for i in range(10))
        
        tracemalloc.start()
        result = self._measure_single_query("Large Query Memory", large_query)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result.memory_used = peak / 1024 / 1024  # Convert to MB
        results.append(result)
        
        print(f"      Large query memory: {result.memory_used:.2f} MB")
        
        total_time = sum(r.parse_time + r.plan_time for r in results)
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        avg_parse = sum(r.parse_time for r in results) / len(results)
        avg_plan = sum(r.plan_time for r in results) / len(results)
        peak_memory = max(r.memory_used for r in results)
        
        return TestSuite("Memory", results, total_time, success_rate, avg_parse, avg_plan, peak_memory)
    
    def _run_scalability_tests(self) -> TestSuite:
        """Test how performance scales with query complexity"""
        print("\n📈 Running Scalability Tests...")
        
        results = []
        
        # Test scalability with increasing variable length
        print("   Testing variable length scalability...")
        for max_length in [1, 2, 5, 10, 20, 50]:
            query = f"MATCH (a)-[*1..{max_length}]->(b) RETURN a, b"
            result = self._measure_single_query(f"VarLen_{max_length}", query)
            results.append(result)
            print(f"      Length {max_length:2d}: {result.parse_time:6.2f}ms")
        
        # Test scalability with increasing pattern complexity
        print("   Testing pattern complexity scalability...")
        for pattern_count in [1, 5, 10, 20, 50]:
            patterns = []
            for i in range(pattern_count):
                patterns.append(f"(n{i}:Type{i})-[r{i}:REL{i}]->(m{i}:Type{i})")
            
            query = "MATCH " + ", ".join(patterns) + " RETURN " + ", ".join(f"n{i}" for i in range(min(pattern_count, 5)))
            result = self._measure_single_query(f"Patterns_{pattern_count}", query)
            results.append(result)
            print(f"      Patterns {pattern_count:2d}: {result.parse_time:6.2f}ms")
        
        # Test scalability with increasing WHERE complexity
        print("   Testing WHERE clause scalability...")
        for condition_count in [1, 5, 10, 20, 50]:
            conditions = []
            for i in range(condition_count):
                conditions.append(f"n.prop{i} > {i}")
            
            query = f"MATCH (n) WHERE {' AND '.join(conditions)} RETURN n"
            result = self._measure_single_query(f"Conditions_{condition_count}", query)
            results.append(result)
            print(f"      Conditions {condition_count:2d}: {result.parse_time:6.2f}ms")
        
        total_time = sum(r.parse_time + r.plan_time for r in results)
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        avg_parse = sum(r.parse_time for r in results) / len(results)
        avg_plan = sum(r.plan_time for r in results) / len(results)
        peak_memory = max(r.memory_used for r in results)
        
        return TestSuite("Scalability", results, total_time, success_rate, avg_parse, avg_plan, peak_memory)
    
    def _measure_single_query(self, test_name: str, query: str, expect_error: bool = False, silent: bool = False) -> TestResult:
        """Measure parsing time and memory for a single query"""
        if not IMPORTS_AVAILABLE:
            return TestResult(test_name, query, False, 0, 0, 0, "Imports not available")
        
        # Measure memory before
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure parsing time
        parse_start = time.perf_counter()
        parse_success = False
        parse_error = ""
        warnings = []
        
        try:
            ast = self.parser.parse(query)
            parse_success = True
        except Exception as e:
            parse_error = str(e)
            if not expect_error and not silent:
                # Get detailed error info
                try:
                    error_details = get_cypher_errors(query)
                    if error_details.get('warnings'):
                        warnings = [w['message'] for w in error_details['warnings'][:3]]
                except:
                    pass
        
        parse_time = (time.perf_counter() - parse_start) * 1000  # Convert to ms
        
        # Measure planning time
        plan_start = time.perf_counter()
        plan_success = False
        
        if parse_success:
            try:
                plan = self.planner.plan(ast)
                plan_success = True
            except Exception as e:
                if not parse_error:
                    parse_error = f"Planning failed: {str(e)}"
        
        plan_time = (time.perf_counter() - plan_start) * 1000  # Convert to ms
        
        # Measure memory after
        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = mem_after - mem_before
        
        success = parse_success and (plan_success or expect_error)
        
        return TestResult(test_name, query, success, parse_time, plan_time, 
                         memory_used, parse_error, warnings)
    
    def _generate_comprehensive_report(self, test_suites: Dict[str, TestSuite]) -> None:
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        # Overall summary
        total_tests = sum(len(suite.results) for suite in test_suites.values())
        total_successes = sum(sum(1 for r in suite.results if r.success) for suite in test_suites.values())
        overall_success_rate = (total_successes / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {total_successes}")
        print(f"   Success Rate: {overall_success_rate:.1f}%")
        print(f"   Test Suites: {len(test_suites)}")
        
        # Performance summary
        all_parse_times = []
        all_plan_times = []
        all_memory_usage = []
        
        for suite in test_suites.values():
            for result in suite.results:
                if result.success:
                    all_parse_times.append(result.parse_time)
                    all_plan_times.append(result.plan_time)
                    all_memory_usage.append(result.memory_used)
        
        if all_parse_times:
            print(f"\n⚡ PERFORMANCE SUMMARY:")
            print(f"   Average Parse Time: {sum(all_parse_times)/len(all_parse_times):.2f}ms")
            print(f"   Fastest Parse: {min(all_parse_times):.2f}ms")
            print(f"   Slowest Parse: {max(all_parse_times):.2f}ms")
            print(f"   Average Plan Time: {sum(all_plan_times)/len(all_plan_times):.2f}ms")
            print(f"   Peak Memory Usage: {max(all_memory_usage):.2f}MB")
        
        # Suite-by-suite breakdown
        print(f"\n📋 SUITE BREAKDOWN:")
        print(f"{'Suite':<15} {'Tests':<8} {'Success':<8} {'Avg Parse':<12} {'Avg Plan':<12} {'Peak Mem':<10}")
        print("-" * 75)
        
        for suite_name, suite in test_suites.items():
            success_count = sum(1 for r in suite.results if r.success)
            print(f"{suite_name:<15} {len(suite.results):<8} {success_count:<8} "
                  f"{suite.avg_parse_time:<12.2f} {suite.avg_plan_time:<12.2f} {suite.peak_memory:<10.2f}")
        
        # Issues analysis
        print(f"\n🔍 ISSUES ANALYSIS:")
        
        # Collect all errors
        all_errors = defaultdict(int)
        slow_queries = []
        memory_intensive = []
        
        for suite_name, suite in test_suites.items():
            for result in suite.results:
                if not result.success and result.error_message:
                    # Categorize error
                    error_msg = result.error_message.lower()
                    if "syntax" in error_msg or "token" in error_msg:
                        all_errors["Syntax Errors"] += 1
                    elif "undefined" in error_msg or "semantic" in error_msg:
                        all_errors["Semantic Errors"] += 1
                    elif "memory" in error_msg:
                        all_errors["Memory Errors"] += 1
                    elif "timeout" in error_msg:
                        all_errors["Timeout Errors"] += 1
                    else:
                        all_errors["Other Errors"] += 1
                
                # Flag slow queries (>100ms parse time)
                if result.parse_time > 100:
                    slow_queries.append((result.test_name, result.parse_time, result.query[:100]))
                
                # Flag memory intensive queries (>10MB)
                if result.memory_used > 10:
                    memory_intensive.append((result.test_name, result.memory_used, result.query[:100]))
        
        if all_errors:
            print("   Error Categories:")
            for error_type, count in all_errors.items():
                print(f"     {error_type}: {count}")
        else:
            print("   ✅ No errors detected!")
        
        if slow_queries:
            print(f"\n   ⚠️ Slow Queries (>{100}ms):")
            for name, time_ms, query_preview in slow_queries[:5]:  # Show top 5
                print(f"     {name}: {time_ms:.2f}ms - {query_preview}...")
        
        if memory_intensive:
            print(f"\n   🧠 Memory Intensive Queries (>10MB):")
            for name, memory_mb, query_preview in memory_intensive[:5]:  # Show top 5
                print(f"     {name}: {memory_mb:.2f}MB - {query_preview}...")
        
        # Performance trends
        print(f"\n📈 PERFORMANCE TRENDS:")
        self._analyze_performance_trends(test_suites)
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        self._generate_recommendations(test_suites, all_errors, slow_queries, memory_intensive)
        
        # Save detailed report to file
        self._save_detailed_report(test_suites, "test_report.json")
        
        print(f"\n📄 Detailed report saved to: test_report.json")
        print("=" * 80)
    
    def _analyze_performance_trends(self, test_suites: Dict[str, TestSuite]) -> None:
        """Analyze performance trends across test suites"""
        
        # Analyze scalability trends
        if "scalability" in test_suites:
            scalability_suite = test_suites["scalability"]
            
            # Group by test type
            varlen_times = []
            pattern_times = []
            condition_times = []
            
            for result in scalability_suite.results:
                if result.test_name.startswith("VarLen_") and result.success:
                    length = int(result.test_name.split("_")[1])
                    varlen_times.append((length, result.parse_time))
                elif result.test_name.startswith("Patterns_") and result.success:
                    count = int(result.test_name.split("_")[1])
                    pattern_times.append((count, result.parse_time))
                elif result.test_name.startswith("Conditions_") and result.success:
                    count = int(result.test_name.split("_")[1])
                    condition_times.append((count, result.parse_time))
            
            # Analyze trends
            if len(varlen_times) > 1:
                varlen_times.sort()
                growth_rate = varlen_times[-1][1] / varlen_times[0][1] if varlen_times[0][1] > 0 else 0
                print(f"   Variable Length Scaling: {growth_rate:.1f}x slowdown from {varlen_times[0][0]} to {varlen_times[-1][0]}")
            
            if len(pattern_times) > 1:
                pattern_times.sort()
                growth_rate = pattern_times[-1][1] / pattern_times[0][1] if pattern_times[0][1] > 0 else 0
                print(f"   Pattern Count Scaling: {growth_rate:.1f}x slowdown from {pattern_times[0][0]} to {pattern_times[-1][0]} patterns")
        
        # Analyze error handling performance
        if "error_handling" in test_suites:
            error_suite = test_suites["error_handling"]
            error_times = [r.parse_time for r in error_suite.results if not r.success]
            if error_times:
                avg_error_time = sum(error_times) / len(error_times)
                print(f"   Error Detection Time: {avg_error_time:.2f}ms average")
    
    def _generate_recommendations(self, test_suites: Dict[str, TestSuite], 
                                all_errors: Dict[str, int], slow_queries: List, 
                                memory_intensive: List) -> None:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance recommendations
        if slow_queries:
            recommendations.append("🐌 Consider optimizing tokenization for complex queries")
            recommendations.append("⚡ Implement query complexity limits to prevent performance issues")
        
        # Memory recommendations
        if memory_intensive:
            recommendations.append("🧠 Implement memory pooling for large query parsing")
            recommendations.append("♻️ Add memory cleanup after parsing large queries")
        
        # Error handling recommendations
        if all_errors.get("Syntax Errors", 0) > 5:
            recommendations.append("🔧 Improve error recovery in tokenizer")
        
        if all_errors.get("Memory Errors", 0) > 0:
            recommendations.append("💾 Add memory limits and graceful degradation")
        
        # Success rate recommendations
        overall_success = sum(len([r for r in suite.results if r.success]) for suite in test_suites.values())
        total_tests = sum(len(suite.results) for suite in test_suites.values())
        success_rate = (overall_success / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate < 90:
            recommendations.append("❗ Investigate and fix failing test cases")
        elif success_rate < 95:
            recommendations.append("⚠️ Monitor edge cases that cause parsing failures")
        
        # Scalability recommendations
        if "scalability" in test_suites:
            scalability_suite = test_suites["scalability"]
            max_parse_time = max(r.parse_time for r in scalability_suite.results if r.success)
            if max_parse_time > 1000:  # > 1 second
                recommendations.append("🚀 Consider implementing incremental parsing for very large queries")
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("   ✅ No major issues detected - performance looks good!")
    
    def _save_detailed_report(self, test_suites: Dict[str, TestSuite], filename: str) -> None:
        """Save detailed test results to JSON file"""
        
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3)
            },
            "test_suites": {}
        }
        
        for suite_name, suite in test_suites.items():
            suite_data = {
                "name": suite.name,
                "total_time": suite.total_time,
                "success_rate": suite.success_rate,
                "avg_parse_time": suite.avg_parse_time,
                "avg_plan_time": suite.avg_plan_time,
                "peak_memory": suite.peak_memory,
                "results": []
            }
            
            for result in suite.results:
                result_data = {
                    "test_name": result.test_name,
                    "query": result.query[:200] + "..." if len(result.query) > 200 else result.query,
                    "success": result.success,
                    "parse_time": result.parse_time,
                    "plan_time": result.plan_time,
                    "memory_used": result.memory_used,
                    "error_message": result.error_message,
                    "warnings": result.warnings
                }
                suite_data["results"].append(result_data)
            
            report_data["test_suites"][suite_name] = suite_data
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
        except Exception as e:
            print(f"   Warning: Could not save report to {filename}: {e}")

def interactive_test_mode():
    """Interactive mode for testing specific queries"""
    print("\n🎮 Interactive Test Mode")
    print("Enter Cypher queries to test (type 'quit' to exit)")
    print("Commands: 'help', 'stats', 'benchmark', 'memory'")
    print("-" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run interactive mode - imports failed")
        return
    
    test_suite = CypherTestSuite()
    session_results = []
    
    while True:
        try:
            query = input("\ncypher> ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'help':
                print("\nCommands:")
                print("  help      - Show this help")
                print("  stats     - Show session statistics")
                print("  benchmark - Run quick benchmark")
                print("  memory    - Show memory usage")
                print("  quit      - Exit")
                continue
            elif query.lower() == 'stats':
                if session_results:
                    successful = sum(1 for r in session_results if r.success)
                    avg_parse = sum(r.parse_time for r in session_results) / len(session_results)
                    max_parse = max(r.parse_time for r in session_results)
                    print(f"\n📊 Session Stats:")
                    print(f"   Queries tested: {len(session_results)}")
                    print(f"   Successful: {successful}")
                    print(f"   Success rate: {successful/len(session_results)*100:.1f}%")
                    print(f"   Average parse time: {avg_parse:.2f}ms")
                    print(f"   Slowest query: {max_parse:.2f}ms")
                else:
                    print("   No queries tested yet")
                continue
            elif query.lower() == 'benchmark':
                print("\n⚡ Quick Benchmark:")
                benchmark_queries = [
                    "MATCH (n) RETURN n",
                    "MATCH (a)-[r]->(b) RETURN a, b",
                    "MATCH (a)-[*1..3]->(b) WHERE a.age > 25 RETURN a, b"
                ]
                for i, bq in enumerate(benchmark_queries, 1):
                    result = test_suite._measure_single_query(f"Bench_{i}", bq, silent=True)
                    status = "✅" if result.success else "❌"
                    print(f"   {status} Query {i}: {result.parse_time:.2f}ms")
                continue
            elif query.lower() == 'memory':
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"\n🧠 Current memory usage: {memory_mb:.2f} MB")
                continue
            
            # Test the query
            result = test_suite._measure_single_query("Interactive", query)
            session_results.append(result)
            
            # Display results
            if result.success:
                print(f"✅ Success! Parse: {result.parse_time:.2f}ms, Plan: {result.plan_time:.2f}ms")
                if result.memory_used > 1:
                    print(f"   Memory: {result.memory_used:.2f}MB")
                if result.warnings:
                    print(f"   Warnings: {', '.join(result.warnings[:2])}")
            else:
                print(f"❌ Failed! Parse: {result.parse_time:.2f}ms")
                print(f"   Error: {result.error_message[:100]}...")
                
                # Try to get more detailed error info
                try:
                    error_details = get_cypher_errors(query)
                    if error_details.get('errors'):
                        for error in error_details['errors'][:2]:
                            print(f"   • {error['code']}: {error['message']}")
                            if error['suggestion']:
                                print(f"     💡 {error['suggestion']}")
                except:
                    pass
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    # Show final session stats
    if session_results:
        successful = sum(1 for r in session_results if r.success)
        print(f"\n📊 Final Session Stats:")
        print(f"   Queries tested: {len(session_results)}")
        print(f"   Successful: {successful} ({successful/len(session_results)*100:.1f}%)")

def main():
    """Main entry point"""
    print("🧪 Cypher Parser Comprehensive Test Suite")
    print("=" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run tests - cypher_planner imports failed")
        print("Make sure you're in the cypher_planner project directory")
        return
    
    print("Choose test mode:")
    print("1. Run full test suite (default)")
    print("2. Interactive testing mode")
    print("3. Quick performance check")
    print("4. Error handling focus")
    print("5. Memory analysis")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        test_suite = CypherTestSuite()
        
        if choice == "" or choice == "1":
            # Full test suite
            results = test_suite.run_all_tests()
        elif choice == "2":
            # Interactive mode
            interactive_test_mode()
            return
        elif choice == "3":
            # Quick performance check
            print("\n⚡ Quick Performance Check")
            results = {
                'performance': test_suite._run_performance_tests(),
                'regression': test_suite._run_regression_tests()
            }
            test_suite._generate_comprehensive_report(results)
        elif choice == "4":
            # Error handling focus
            print("\n🔍 Error Handling Analysis")
            results = {
                'error_handling': test_suite._run_error_handling_tests(),
                'edge_cases': test_suite._run_edge_case_tests()
            }
            test_suite._generate_comprehensive_report(results)
        elif choice == "5":
            # Memory analysis
            print("\n🧠 Memory Analysis")
            results = {
                'memory': test_suite._run_memory_tests(),
                'stress': test_suite._run_stress_tests()
            }
            test_suite._generate_comprehensive_report(results)
        else:
            print("❌ Invalid choice")
            return
        
    except KeyboardInterrupt:
        print("\n\n👋 Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()