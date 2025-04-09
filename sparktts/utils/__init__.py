import threading


class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        # 使用 RLock 可重入锁，避免死锁
        self._lock = threading.RLock()

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def set(self, key, value):
        with self._lock:
            self._dict[key] = value

    def pop(self, key):
        with self._lock:
            return self._dict.pop(key, None)
