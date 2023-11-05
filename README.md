# Seek ML

An online feature store serving pattern that uses `voyager` and is completely in-memory.

Usage:

```py
from seek_ml.seek_store import SeekStore


store = SeekStore()
group: str = "group"
store.add(group, 2, np.arange(10))
store.add(group, 200, np.arange(10) + 1)

store.fetch(group, 2)
```

Since `voyage` provides both a simple key-value lookup and a vector store, we can provide a
simple abstraction to provide an online feature store. The rough idea is that at any point in time:

* push features to a particular group by entity id via `store.add(group, id, vector)`
* retrieve features by group and entity id via `store.fetch(group, id)`

This can help provide ways to access features in a multi-entity setup or in a setup where
each event generates different set of features.

This repository is also a simple exemplar repository with black, isort, ruff and pyright which
I want to iterate and use as a template. 

**Dev Scripts**

```
just format  # format
just lint  # linting and static analysis
just test  # running tests
```