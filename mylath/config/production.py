# Example configuration file
# config/production.py
"""Production configuration for MyLath"""

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'max_connections': 100,
    'socket_keepalive': True,
    'socket_keepalive_options': {
        'TCP_KEEPIDLE': 1,
        'TCP_KEEPINTVL': 3,
        'TCP_KEEPCNT': 5,
    },
    'retry_on_timeout': True,
    'health_check_interval': 30
}

# Replication setup for high availability
REPLICA_CONFIGS = [
    {'host': 'replica1.example.com', 'port': 6379, 'db': 0},
    {'host': 'replica2.example.com', 'port': 6379, 'db': 0},
]

# Vector index configuration
VECTOR_CONFIG = {
    'm': 16,                    # Max connections per node in HNSW
    'ef_construction': 128,     # Size of dynamic candidate list during construction
    'ef_search': 768,          # Size of dynamic candidate list during search
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'threaded': True
}

# Authentication
AUTH_CONFIG = {
    'secret_key': 'your-secret-key-here',  # Use environment variable in production
    'algorithm': 'HS256',
    'token_expiry': 3600  # 1 hour
}

# Monitoring
MONITORING_CONFIG = {
    'enabled': True,
    'dashboard_port': 8080,
    'metrics_retention': 86400  # 24 hours
}

# Performance tuning
PERFORMANCE_CONFIG = {
    'cache_ttl': 3600,
    'batch_size': 1000,
    'connection_pool_size': 50,
    'query_timeout': 30
}
