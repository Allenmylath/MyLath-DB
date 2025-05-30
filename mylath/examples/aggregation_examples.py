# mylath/examples/aggregation_examples.py
"""
Comprehensive examples of MyLath's aggregation capabilities
"""

from mylath import Graph, RedisStorage
import random
import time


def setup_sample_data(graph):
    """Create sample data for aggregation examples"""
    print("Setting up sample data...")
    
    # Companies
    companies = []
    for company_name in ["TechCorp", "DataSoft", "AILabs", "CloudCo", "DevOps Inc"]:
        company = graph.create_node("company", {
            "name": company_name,
            "industry": random.choice(["Tech", "AI", "Cloud", "DevOps"]),
            "size": random.choice(["Small", "Medium", "Large"]),
            "revenue": random.randint(1_000_000, 100_000_000),
            "founded": random.randint(2000, 2020)
        })
        companies.append(company)
    
    # People
    cities = ["NYC", "SF", "LA", "Chicago", "Boston", "Seattle", "Austin"]
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance"]
    
    people = []
    for i in range(100):
        person = graph.create_node("person", {
            "name": f"Person_{i}",
            "age": random.randint(22, 65),
            "salary": random.randint(50_000, 200_000),
            "city": random.choice(cities),
            "department": random.choice(departments),
            "experience_years": random.randint(0, 20),
            "performance_score": round(random.uniform(1.0, 5.0), 2)
        })
        people.append(person)
        
        # Assign to company
        company = random.choice(companies)
        graph.create_edge("works_at", person.id, company.id, {
            "start_date": f"2020-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "role": random.choice(["Junior", "Senior", "Lead", "Manager"])
        })
    
    # Create some friendships
    for i in range(200):
        person1 = random.choice(people)
        person2 = random.choice(people)
        if person1.id != person2.id:
            graph.create_edge("knows", person1.id, person2.id, {
                "closeness": random.randint(1, 10),
                "since": f"20{random.randint(15, 23)}"
            })
    
    # Projects
    for i in range(30):
        project = graph.create_node("project", {
            "name": f"Project_{i}",
            "budget": random.randint(10_000, 1_000_000),
            "status": random.choice(["Planning", "Active", "Completed", "On Hold"]),
            "priority": random.choice(["Low", "Medium", "High", "Critical"])
        })
        
        # Assign people to projects
        num_people = random.randint(2, 8)
        project_people = random.sample(people, num_people)
        for person in project_people:
            graph.create_edge("assigned_to", person.id, project.id, {
                "hours_per_week": random.randint(10, 40),
                "role": random.choice(["Developer", "Analyst", "Lead", "Tester"])
            })
    
    print(f"Created {len(people)} people, {len(companies)} companies, 30 projects")
    return people, companies


def basic_aggregation_examples(graph):
    """Basic aggregation examples"""
    print("\n=== BASIC AGGREGATIONS ===")
    
    # Simple counts
    print("\n1. Basic Counts:")
    person_count = graph.V().has("label", "person").count()
    print(f"Total people: {person_count}")
    
    company_count = graph.V().has("label", "company").count()
    print(f"Total companies: {company_count}")
    
    # Count by property
    print("\n2. Count by City:")
    city_counts = graph.V().has("label", "person").count_by("city")
    for city, count in sorted(city_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {city}: {count} people")
    
    # Simple statistics
    print("\n3. Salary Statistics:")
    people = graph.V().has("label", "person")
    avg_salary = people.avg("salary")
    min_salary = people.min("salary")
    max_salary = people.max("salary")
    
    print(f"  Average salary: ${avg_salary:,.2f}")
    print(f"  Min salary: ${min_salary:,}")
    print(f"  Max salary: ${max_salary:,}")


def advanced_aggregation_examples(graph):
    """Advanced aggregation with group by and multiple metrics"""
    print("\n=== ADVANCED AGGREGATIONS ===")
    
    # Group by single property with multiple aggregations
    print("\n1. Department Analysis:")
    dept_analysis = (graph.V().has("label", "person")
                    .group_by("department")
                    .count("headcount")
                    .avg("salary", "avg_salary")
                    .avg("age", "avg_age")
                    .avg("performance_score", "avg_performance")
                    .sum("salary", "total_payroll")
                    .execute())
    
    for dept, metrics in dept_analysis.groups.items():
        print(f"\n  {dept}:")
        print(f"    Headcount: {metrics['headcount']}")
        print(f"    Avg Salary: ${metrics['avg_salary']:,.2f}")
        print(f"    Avg Age: {metrics['avg_age']:.1f}")
        print(f"    Avg Performance: {metrics['avg_performance']:.2f}")
        print(f"    Total Payroll: ${metrics['total_payroll']:,.2f}")
    
    # Group by multiple properties
    print("\n2. City + Department Analysis:")
    city_dept_analysis = (graph.V().has("label", "person")
                         .group_by("city", "department")
                         .count("headcount")
                         .avg("salary", "avg_salary")
                         .having(lambda g: g["headcount"] >= 2)  # Only groups with 2+ people
                         .order_by("avg_salary")
                         .limit(10)
                         .execute())
    
    for (city, dept), metrics in city_dept_analysis.groups.items():
        print(f"  {city} - {dept}: {metrics['headcount']} people, avg salary ${metrics['avg_salary']:,.2f}")


def company_analysis_examples(graph):
    """Company-focused aggregations"""
    print("\n=== COMPANY ANALYSIS ===")
    
    # Company size analysis
    print("\n1. Company Employee Count:")
    company_headcount = (graph.V().has("label", "company")
                        .group_by_func(lambda company: company.properties["name"])
                        .count("employee_count")
                        .execute())
    
    # Get actual headcount by traversing to employees
    print("\n2. Actual Employee Count (via traversal):")
    for company in graph.V().has("label", "company").to_list():
        employee_count = graph.V(company.id).in_("works_at").count()
        print(f"  {company.properties['name']}: {employee_count} employees")
    
    # Company performance metrics
    print("\n3. Company Performance Analysis:")
    for company in graph.V().has("label", "company").to_list():
        employees = graph.V(company.id).in_("works_at")
        
        if employees.count() > 0:
            avg_performance = employees.avg("performance_score")
            avg_salary = employees.avg("salary")
            total_payroll = employees.sum("salary")
            
            print(f"\n  {company.properties['name']}:")
            print(f"    Revenue: ${company.properties['revenue']:,}")
            print(f"    Employees: {employees.count()}")
            print(f"    Avg Performance: {avg_performance:.2f}")
            print(f"    Avg Salary: ${avg_salary:,.2f}")
            print(f"    Total Payroll: ${total_payroll:,.2f}")
            print(f"    Revenue per Employee: ${company.properties['revenue'] / employees.count():,.2f}")


def complex_traversal_aggregations(graph):
    """Complex aggregations involving graph traversals"""
    print("\n=== COMPLEX TRAVERSAL AGGREGATIONS ===")
    
    # Friend network analysis
    print("\n1. Social Network Analysis:")
    social_stats = (graph.V().has("label", "person")
                   .group_by_func(lambda person: 
                       graph.V(person.id).out("knows").count())  # Group by friend count
                   .count("people_count")
                   .execute())
    
    print("  Friend Count Distribution:")
    for friend_count, stats in sorted(social_stats.groups.items()):
        print(f"    {friend_count} friends: {stats['people_count']} people")
    
    # Project workload analysis
    print("\n2. Project Workload Analysis:")
    for person in graph.V().has("label", "person").limit(5).to_list():
        projects = graph.V(person.id).out("assigned_to")
        project_count = projects.count()
        
        if project_count > 0:
            total_hours = projects.outE("assigned_to").sum("hours_per_week")
            project_names = projects.collect_values("name")
            
            print(f"\n  {person.properties['name']}:")
            print(f"    Projects: {project_count}")
            print(f"    Total Hours/Week: {total_hours}")
            print(f"    Projects: {', '.join(project_names)}")
    
    # Cross-company collaboration
    print("\n3. Cross-Company Friendships:")
    cross_company_friends = 0
    same_company_friends = 0
    
    for person in graph.V().has("label", "person").to_list():
        person_company = graph.V(person.id).out("works_at").to_list()
        if not person_company:
            continue
        person_company = person_company[0]
        
        friends = graph.V(person.id).out("knows").to_list()
        for friend in friends:
            friend_company = graph.V(friend.id).out("works_at").to_list()
            if friend_company:
                if friend_company[0].id == person_company.id:
                    same_company_friends += 1
                else:
                    cross_company_friends += 1
    
    print(f"  Same company friendships: {same_company_friends}")
    print(f"  Cross company friendships: {cross_company_friends}")
    total = same_company_friends + cross_company_friends
    if total > 0:
        print(f"  Cross-company friendship rate: {cross_company_friends/total*100:.1f}%")


def statistical_analysis_examples(graph):
    """Advanced statistical analysis"""
    print("\n=== STATISTICAL ANALYSIS ===")
    
    # Salary distribution analysis
    print("\n1. Salary Distribution Analysis:")
    salary_stats = (graph.V().has("label", "person")
                   .group_by("department")
                   .avg("salary", "avg_salary")
                   .median("salary", "median_salary")
                   .percentile("salary", 0.25, "p25_salary")
                   .percentile("salary", 0.75, "p75_salary")
                   .percentile("salary", 0.90, "p90_salary")
                   .collect("salary", "all_salaries")
                   .execute())
    
    for dept, stats in salary_stats.groups.items():
        print(f"\n  {dept} Salary Distribution:")
        print(f"    Average: ${stats['avg_salary']:,.2f}")
        print(f"    Median: ${stats['median_salary']:,.2f}")
        print(f"    25th percentile: ${stats['p25_salary']:,.2f}")
        print(f"    75th percentile: ${stats['p75_salary']:,.2f}")
        print(f"    90th percentile: ${stats['p90_salary']:,.2f}")
        print(f"    IQR: ${stats['p75_salary'] - stats['p25_salary']:,.2f}")
    
    # Performance correlation analysis
    print("\n2. Performance vs Experience Analysis:")
    performance_by_exp = (graph.V().has("label", "person")
                         .group_by_func(lambda p: min(p.properties.get("experience_years", 0) // 5 * 5, 15))  # Group by 5-year bands
                         .avg("performance_score", "avg_performance")
                         .avg("salary", "avg_salary")
                         .count("count")
                         .execute())
    
    print("  Experience Band | Avg Performance | Avg Salary | Count")
    print("  " + "-" * 55)
    for exp_band, stats in sorted(performance_by_exp.groups.items()):
        if isinstance(exp_band, (int, float)):
            band_label = f"{exp_band}-{exp_band+4} years"
        else:
            band_label = str(exp_band)
        print(f"  {band_label:14} | {stats['avg_performance']:13.2f} | ${stats['avg_salary']:9,.0f} | {stats['count']:5}")


def time_based_aggregations(graph):
    """Time-based aggregation examples"""
    print("\n=== TIME-BASED ANALYSIS ===")
    
    # Hiring patterns by year (from work start dates)
    print("\n1. Hiring Patterns:")
    hiring_by_year = {}
    
    for edge in graph.E().has("label", "works_at").to_list():
        start_date = edge.properties.get("start_date", "")
        if start_date:
            year = start_date.split("-")[0]
            hiring_by_year[year] = hiring_by_year.get(year, 0) + 1
    
    print("  Year | Hires")
    print("  " + "-" * 12)
    for year in sorted(hiring_by_year.keys()):
        print(f"  {year} | {hiring_by_year[year]:5}")
    
    # Age distribution analysis
    print("\n2. Age Distribution:")
    age_distribution = (graph.V().has("label", "person")
                       .group_by_func(lambda p: f"{p.properties.get('age', 0) // 10 * 10}-{p.properties.get('age', 0) // 10 * 10 + 9}")
                       .count("count")
                       .avg("salary", "avg_salary")
                       .execute())
    
    print("  Age Range | Count | Avg Salary")
    print("  " + "-" * 30)
    for age_range, stats in sorted(age_distribution.groups.items()):
        print(f"  {age_range:9} | {stats['count']:5} | ${stats['avg_salary']:9,.0f}")


def project_analytics_examples(graph):
    """Project-focused analytics"""
    print("\n=== PROJECT ANALYTICS ===")
    
    # Project status analysis
    print("\n1. Project Status Overview:")
    project_stats = (graph.V().has("label", "project")
                    .group_by("status")
                    .count("project_count")
                    .sum("budget", "total_budget")
                    .avg("budget", "avg_budget")
                    .execute())
    
    for status, stats in project_stats.groups.items():
        print(f"\n  {status} Projects:")
        print(f"    Count: {stats['project_count']}")
        print(f"    Total Budget: ${stats['total_budget']:,.2f}")
        print(f"    Avg Budget: ${stats['avg_budget']:,.2f}")
    
    # Resource allocation analysis
    print("\n2. Resource Allocation by Priority:")
    priority_analysis = (graph.V().has("label", "project")
                        .group_by("priority")
                        .count("project_count")
                        .sum("budget", "total_budget")
                        .execute())
    
    total_budget = sum(stats["total_budget"] for stats in priority_analysis.groups.values())
    
    for priority, stats in priority_analysis.groups.items():
        budget_pct = (stats["total_budget"] / total_budget * 100) if total_budget > 0 else 0
        print(f"  {priority:8} | {stats['project_count']:3} projects | ${stats['total_budget']:10,.0f} ({budget_pct:5.1f}%)")
    
    # Team size analysis
    print("\n3. Project Team Size Analysis:")
    team_sizes = {}
    for project in graph.V().has("label", "project").to_list():
        team_size = graph.V(project.id).in_("assigned_to").count()
        team_sizes[project.properties["name"]] = {
            "team_size": team_size,
            "budget": project.properties["budget"],
            "status": project.properties["status"],
            "priority": project.properties["priority"]
        }
    
    # Analyze budget per team member
    print("\n  Budget Efficiency (Budget per Team Member):")
    efficiency_data = []
    for project_name, data in team_sizes.items():
        if data["team_size"] > 0:
            efficiency = data["budget"] / data["team_size"]
            efficiency_data.append((project_name, efficiency, data))
    
    # Sort by efficiency (highest budget per person)
    efficiency_data.sort(key=lambda x: x[1], reverse=True)
    
    print("  Project | Budget/Person | Team Size | Status")
    print("  " + "-" * 50)
    for project_name, efficiency, data in efficiency_data[:10]:  # Top 10
        print(f"  {project_name[:15]:15} | ${efficiency:11,.0f} | {data['team_size']:9} | {data['status']}")


def custom_aggregation_examples(graph):
    """Examples of custom aggregation functions"""
    print("\n=== CUSTOM AGGREGATIONS ===")
    
    # Custom grouping function - performance bands
    print("\n1. Performance Bands Analysis:")
    def performance_band(person):
        score = person.properties.get("performance_score", 0)
        if score >= 4.5:
            return "Excellent (4.5+)"
        elif score >= 4.0:
            return "Good (4.0-4.5)"
        elif score >= 3.0:
            return "Average (3.0-4.0)"
        else:
            return "Needs Improvement (<3.0)"
    
    performance_analysis = (graph.V().has("label", "person")
                           .group_by_func(performance_band)
                           .count("count")
                           .avg("salary", "avg_salary")
                           .collect("name", "employees", unique=False)
                           .execute())
    
    for band, stats in performance_analysis.groups.items():
        print(f"\n  {band}:")
        print(f"    Count: {stats['count']} people")
        print(f"    Avg Salary: ${stats['avg_salary']:,.2f}")
        print(f"    Sample Names: {', '.join(stats['employees'][:3])}")
    
    # Custom having clause - high-value departments
    print("\n2. High-Value Departments (avg salary > $100k):")
    high_value_depts = (graph.V().has("label", "person")
                       .group_by("department")
                       .count("headcount")
                       .avg("salary", "avg_salary")
                       .sum("salary", "total_payroll")
                       .having(lambda g: g["avg_salary"] > 100000)
                       .order_by("avg_salary")
                       .execute())
    
    for dept, stats in high_value_depts.groups.items():
        print(f"  {dept}: {stats['headcount']} people, avg ${stats['avg_salary']:,.2f}")
    
    # Multi-level analysis
    print("\n3. Company Efficiency Analysis:")
    for company in graph.V().has("label", "company").to_list():
        employees = graph.V(company.id).in_("works_at").to_list()
        if not employees:
            continue
            
        # Department breakdown within company
        dept_breakdown = {}
        total_payroll = 0
        
        for employee in employees:
            dept = employee.properties.get("department", "Unknown")
            salary = employee.properties.get("salary", 0)
            
            if dept not in dept_breakdown:
                dept_breakdown[dept] = {"count": 0, "total_salary": 0}
            
            dept_breakdown[dept]["count"] += 1
            dept_breakdown[dept]["total_salary"] += salary
            total_payroll += salary
        
        revenue = company.properties.get("revenue", 0)
        profit_margin = ((revenue - total_payroll) / revenue * 100) if revenue > 0 else 0
        
        print(f"\n  {company.properties['name']}:")
        print(f"    Revenue: ${revenue:,}")
        print(f"    Total Payroll: ${total_payroll:,}")
        print(f"    Estimated Profit Margin: {profit_margin:.1f}%")
        print(f"    Revenue per Employee: ${revenue / len(employees):,.0f}")
        
        for dept, data in dept_breakdown.items():
            avg_salary = data["total_salary"] / data["count"]
            payroll_pct = (data["total_salary"] / total_payroll * 100) if total_payroll > 0 else 0
            print(f"      {dept}: {data['count']} people, avg ${avg_salary:,.0f} ({payroll_pct:.1f}% of payroll)")


def benchmark_aggregations(graph):
    """Benchmark aggregation performance"""
    print("\n=== AGGREGATION PERFORMANCE ===")
    
    import time
    
    # Benchmark different aggregation operations
    operations = [
        ("Simple count", lambda: graph.V().has("label", "person").count()),
        ("Group by city", lambda: graph.V().has("label", "person").count_by("city")),
        ("Complex aggregation", lambda: graph.V().has("label", "person")
                                              .group_by("department")
                                              .count("count")
                                              .avg("salary", "avg_salary")
                                              .sum("salary", "total")
                                              .execute()),
        ("Statistical analysis", lambda: graph.V().has("label", "person")
                                               .group_by("city")
                                               .median("salary", "median")
                                               .percentile("salary", 0.95, "p95")
                                               .execute()),
    ]
    
    print("\nOperation | Time (ms) | Result")
    print("-" * 40)
    
    for name, operation in operations:
        start_time = time.time()
        result = operation()
        elapsed = (time.time() - start_time) * 1000
        
        if hasattr(result, 'total_groups'):
            result_desc = f"{result.total_groups} groups"
        elif isinstance(result, dict):
            result_desc = f"{len(result)} items"
        else:
            result_desc = str(result)
        
        print(f"{name:20} | {elapsed:8.2f} | {result_desc}")


def main():
    """Run all aggregation examples"""
    print("MyLath Aggregation Framework Examples")
    print("=" * 50)
    
    # Initialize
    storage = RedisStorage(db=14)  # Use separate DB for examples
    graph = Graph(storage)
    
    try:
        # Clear any existing data
        storage.redis.flushdb()
        
        # Setup sample data
        people, companies = setup_sample_data(graph)
        
        # Run examples
        basic_aggregation_examples(graph)
        advanced_aggregation_examples(graph)
        company_analysis_examples(graph)
        complex_traversal_aggregations(graph)
        statistical_analysis_examples(graph)
        time_based_aggregations(graph)
        project_analytics_examples(graph)
        custom_aggregation_examples(graph)
        benchmark_aggregations(graph)
        
        print(f"\n=== SUMMARY ===")
        stats = graph.get_stats()
        print(f"Final graph stats: {stats}")
        
    finally:
        # Cleanup
        storage.redis.flushdb()
    
    print("\nAggregation examples completed! 🎉")


if __name__ == "__main__":
    main()
