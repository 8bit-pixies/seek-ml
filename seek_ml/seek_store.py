import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy import ndarray
from voyager import Index, Space


class SeekStore:
    _seek_groups = {}

    def __init__(
        self,
        default_space: Space = Space.Euclidean,
        default_ef_construction: int = 200,
        seek_mapping: Optional[dict] = None,
    ):
        self.default_index_config = {
            "space": default_space,
            "ef_construction": default_ef_construction,
        }
        self._seek_groups: Dict[str, Any] = (
            {} if seek_mapping is None else self._reload_mapping(seek_mapping)
        )

    def _reload_mapping(self, mapping: Dict[str, Path]):  # pragma: no cover
        # Index.load() is legal.
        _seek_groups: Dict[str, Any] = {
            key: Index.load(open(value)) for key, value in mapping.items()
        }
        return _seek_groups

    def add(self, group: str, entity_id: int, vector: ndarray, index_config: dict = {}):
        if group not in self._seek_groups.keys():
            index_config["num_dimensions"] = vector.shape[-1]
            self._seek_groups[group] = Index(
                **{**self.default_index_config, **index_config}
            )
        self._seek_groups[group].add_item(vector=vector, id=entity_id)
        return True

    def add_batch(
        self,
        group: str,
        entity_ids: List[int],
        vectors: ndarray,
        index_config: dict = {},
    ):
        if group not in self._seek_groups:
            index_config["num_dimensions"] = vectors.shape[-1]
            self._seek_groups[group] = Index(
                **{**self.default_index_config, **index_config}
            )
        self._seek_groups[group].add_items(vectors=vectors, ids=entity_ids)
        return True

    def fetch(
        self, group: str, entity_id: int, default_value: Optional[ndarray] = None
    ):
        if group not in self._seek_groups:
            raise KeyError(f"Group: {group} not found in index")
        try:
            return self._seek_groups[group].get_vector(entity_id)
        except RuntimeError:
            return default_value

    def fetch_batch(
        self, group: str, entity_ids: List[int], default_value: Optional[ndarray] = None
    ):
        # note that in batch it will fill the offending item with np.nan
        if group not in self._seek_groups:
            raise KeyError(f"Group: {group} not found in index")
        try:
            return self._seek_groups[group].get_vectors(entity_ids)
        except RuntimeError:
            # try to do it one at a time - will be slow
            vectors = [
                self.fetch(group, entity_id, default_value) for entity_id in entity_ids
            ]
            print(vectors)
            if len([x for x in vectors if x is None]) == len(vectors):
                return None
            else:
                default_vector_size = [x for x in vectors if x is not None][0].shape[-1]
                default_vector = np.empty((default_vector_size,))
                default_vector[:] = np.nan
                vectors = [
                    default_vector if default_vector is not None else default_vector
                    for x in vectors
                ]
            return np.vstack(vectors)

    def load(self, dir_path):
        for path in Path(dir_path).glob("*.voy"):
            key = path.name.removesuffix(".voy")
            self._seek_groups[key] = Index.load(open(path, "rb"))

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        for key in self._seek_groups.keys():
            self._seek_groups[key].save(os.path.join(dir_path, f"{key}.voy"))
