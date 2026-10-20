"""Cholesky re-orthogonalisation with the library of the backend: orthonormal, same column space as QR, dtype kept."""
import numpy as np
import pytest
import functions as fn


def _config(precision):
    class Cfg:
        backend = fn.NumpyBackend(1)
    return Cfg


@pytest.mark.parametrize("dtype,tol", [(np.complex64, 2e-4), (np.complex128, 1e-12)])
def test_numpy_cholesky_orthonormal_same_span_dtype(dtype, tol):
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((5, 24, 6)) + 1j * rng.standard_normal((5, 24, 6))).astype(dtype)
    cfg = _config(None)
    q = fn.reortho_qr(cfg, a, "Cholesky")
    assert q.dtype == dtype
    gram = np.swapaxes(q.conj(), -1, -2) @ q
    assert np.abs(gram - np.eye(6)).max() < tol
    qr = fn.reortho_qr(cfg, a, "QR")
    # same column space: the projector onto the span is identical
    p1 = q @ np.swapaxes(q.conj(), -1, -2)
    p2 = qr @ np.swapaxes(qr.conj(), -1, -2)
    assert np.abs(p1 - p2).max() < 10 * tol


def test_numpy_cholesky_does_not_import_jax_path(monkeypatch):
    called = []
    monkeypatch.setattr(fn, "cholesky_orthonormalize_complex_jax", lambda w: called.append(1))
    a = np.random.default_rng(0).standard_normal((2, 8, 3)).astype(np.complex64)
    fn.reortho_qr(_config(None), a, "Cholesky")
    assert not called
