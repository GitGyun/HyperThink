"""Public artifact helpers; shards are published before merging under a lock."""
import os
import fcntl
from pathlib import Path

def merge_distill_shards(root, name, count, shard):
    root = Path(root)
    parts = root / ".shards" / str(count)
    # Completion markers distinguish closed files from another worker still writing.
    (parts / str(shard) / (name + ".done")).touch()
    with open(parts / (name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paths = [parts / str(i) / name for i in range(count)]
        if not all(Path(str(p) + ".done").exists() for p in paths):
            return
        destination = root / name
        with open(str(destination) + ".tmp", "w") as merged:
            for p in paths:
                merged.write(p.read_text())
        os.replace(str(destination) + ".tmp", destination)
