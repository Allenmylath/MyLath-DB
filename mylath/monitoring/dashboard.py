# mylath/monitoring/dashboard.py
from flask import Flask, render_template, jsonify
import json
import time


class MonitoringDashboard:
    """Web dashboard for monitoring MyLath performance"""
    
    def __init__(self, graph, storage):
        self.app = Flask(__name__)
        self.graph = graph
        self.storage = storage
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/stats')
        def get_stats():
            stats = self.graph.get_stats()
            
            # Add performance metrics
            if hasattr(self.storage, 'metrics'):
                perf_stats = self.storage.metrics.get_all_stats()
                stats['performance'] = perf_stats
            
            # Add memory usage
            try:
                memory_info = self.storage.redis.info('memory')
                stats['memory'] = {
                    'used': memory_info.get('used_memory_human'),
                    'peak': memory_info.get('used_memory_peak_human'),
                    'fragmentation_ratio': memory_info.get('mem_fragmentation_ratio')
                }
            except:
                pass
            
            return jsonify(stats)
        
        @self.app.route('/api/health')
        def health_check():
            try:
                # Test Redis connection
                self.storage.redis.ping()
                
                # Get basic stats
                stats = self.graph.get_stats()
                
                return jsonify({
                    'status': 'healthy',
                    'timestamp': time.time(),
                    'stats': stats
                })
            except Exception as e:
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': time.time()
                }), 500
    
    def run(self, host='localhost', port=8080, debug=False):
        self.app.run(host=host, port=port, debug=debug)
