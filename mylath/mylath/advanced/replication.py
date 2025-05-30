# mylath/advanced/replication.py
import redis
from typing import List, Optional
import threading
import time


class ReplicationManager:
    """Manage read replicas for improved read performance"""
    
    def __init__(self, master_config: dict, replica_configs: List[dict]):
        self.master = redis.Redis(**master_config)
        self.replicas = [redis.Redis(**config) for config in replica_configs]
        self.current_replica = 0
        self.lock = threading.Lock()
        
        # Health check thread
        self._start_health_checks()
    
    def get_read_connection(self) -> redis.Redis:
        """Get a read connection (round-robin through replicas)"""
        if not self.replicas:
            return self.master
        
        with self.lock:
            replica = self.replicas[self.current_replica]
            self.current_replica = (self.current_replica + 1) % len(self.replicas)
            return replica
    
    def get_write_connection(self) -> redis.Redis:
        """Get write connection (always master)"""
        return self.master
    
    def _start_health_checks(self):
        """Start background health checking for replicas"""
        def health_check():
            while True:
                healthy_replicas = []
                for replica in self.replicas:
                    try:
                        replica.ping()
                        healthy_replicas.append(replica)
                    except:
                        pass  # Replica is down
                
                with self.lock:
                    self.replicas = healthy_replicas
                
                time.sleep(30)  # Check every 30 seconds
        
        thread = threading.Thread(target=health_check, daemon=True)
        thread.start()
