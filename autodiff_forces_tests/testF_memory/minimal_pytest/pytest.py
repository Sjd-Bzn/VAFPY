"""Minimal stand-in for the few pytest names the vafpy test modules use at import time (pytest is not installed in
the VSC5 environment). Fixtures are plain functions; run_exact.py calls the tests directly."""
import contextlib
import importlib


class Skipped(Exception):
    pass


def skip(msg=""):
    raise Skipped(msg)


def importorskip(name):
    return importlib.import_module(name)


def fixture(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda f: f


class _Mark:
    def __getattr__(self, name):
        def marker(*a, **k):
            return lambda f: f
        return marker


mark = _Mark()


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"{exc.__name__} not raised")
