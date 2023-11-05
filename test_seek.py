import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from seek_ml.seek_store import SeekStore


@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(dirpath)

    with cd(dirpath, cleanup):
        yield dirpath


def test_add_and_fetch_item():
    store = SeekStore()

    group: str = "group"
    store.add(group, 2, np.arange(10))
    store.add(group, 200, np.arange(10) + 1)

    assert np.array_equal(store.fetch(group, 2), np.arange(10))
    assert not np.array_equal(store.fetch(group, 2), store.fetch(group, 200))
    assert store.fetch(group, 100) is None


def test_add_and_fetch_batch_items():
    store = SeekStore()

    group: str = "group"
    store.add_batch(group, [2, 200], np.arange(20).reshape(2, -1).astype(np.float32))

    assert np.array_equal(
        store.fetch_batch(group, [2, 200]), np.arange(20).reshape(2, -1)
    )
    assert not np.array_equal(store.fetch(group, 2), store.fetch(group, 200))
    assert (
        store.fetch_batch(group, [2, 100]).shape
        == store.fetch_batch(group, [2, 200]).shape
    )
    assert store.fetch_batch(group, [100]) is None


def test_multiple_groups():
    store = SeekStore()

    store.add("group1", 2, np.arange(10))
    store.add("group2", 200, np.arange(20))

    assert np.array_equal(store.fetch("group1", 2), np.arange(10))
    assert np.array_equal(store.fetch("group2", 200), np.arange(20))


def test_raise_error_on_no_group():
    with pytest.raises(KeyError):
        store = SeekStore()
        store.fetch("should_error", 1)


def test_raise_error_on_no_group_batch():
    with pytest.raises(KeyError):
        store = SeekStore()
        store.fetch_batch("should_error", [1])


def test_save_in_temp_dir():
    with tempdir() as dirpath:
        save_dir = Path(dirpath).joinpath("output")
        store = SeekStore()

        group: str = "group"
        store.add(group, 2, np.arange(10))
        store.add(group, 200, np.arange(10) + 1)
        store.save(save_dir)

        store_reload = SeekStore()
        store_reload.load(save_dir)

        assert np.array_equal(store.fetch(group, 2), store_reload.fetch(group, 2))
