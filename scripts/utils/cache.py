from threading import Lock
import logging

logger = logging.getLogger(__name__)

class LRUCache:
    """Least Recently Used Cache implementation"""
    
    def __init__(self, maxsize: int = 100):
        self.cache = {}
        self.maxsize = maxsize
        self.access_order = []
        self.lock = Lock()
    
    def __getitem__(self, key):
        with self.lock:
            if key in self.cache:
                # Move to end to mark as recently used
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            raise KeyError(key)
    
    def __setitem__(self, key, value):
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used item
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
                logger.debug(f"LRU cache evicted: {lru_key}")
            
            self.cache[key] = value
            self.access_order.append(key)
            logger.debug(f"LRU cache set: {key}")
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            logger.info("LRU cache cleared.")