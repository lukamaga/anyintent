"""Label-vector cache: in memory (the last few label sets) and optionally on disk.

The key is the sha256 of a canonical JSON of the model revision, the embedder, the maximum label length, the engine
family, the compute dtype and device type, and the label texts in order.
"""
import collections
import hashlib
import json
import os

import numpy as np


def cache_key(**fields):
    blob = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LabelCache:
    def __init__(self, max_items=8, cache_dir=None):
        self.max_items = int(max_items)
        self.cache_dir = cache_dir
        self.mem = collections.OrderedDict()
        self.hits = {"memory": 0, "disk": 0, "miss": 0}

    def get(self, key, texts):
        if key in self.mem:
            self.mem.move_to_end(key)
            self.hits["memory"] += 1
            return self.mem[key]
        if self.cache_dir:
            npy, side = os.path.join(self.cache_dir, key + ".npy"), os.path.join(self.cache_dir, key + ".json")
            try:
                with open(side, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("texts") == list(texts):
                    arr = np.load(npy)
                    if arr.shape[0] == len(texts):
                        self._put_mem(key, arr)
                        self.hits["disk"] += 1
                        return arr
            except (OSError, ValueError):
                pass
        self.hits["miss"] += 1
        return None

    def _put_mem(self, key, arr):
        self.mem[key] = arr
        self.mem.move_to_end(key)
        while len(self.mem) > self.max_items:
            self.mem.popitem(last=False)

    def put(self, key, texts, arr, fields=None):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._put_mem(key, arr)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            npy, side = os.path.join(self.cache_dir, key + ".npy"), os.path.join(self.cache_dir, key + ".json")
            tmp = npy + f".tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                np.save(f, arr)
            os.replace(tmp, npy)
            tmp = side + f".tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"texts": list(texts), "fields": fields or {}}, f, ensure_ascii=False)
            os.replace(tmp, side)
