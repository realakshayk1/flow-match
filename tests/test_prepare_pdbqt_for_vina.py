"""Unit tests for PDBQT manifest helpers (no obabel required)."""

from pathlib import Path

from scripts.prepare_pdbqt_for_vina import _cache_key, _resolve_paths


def test_resolve_paths_crossdock_columns():
    row = {"receptor_pdb": "/r.pdb", "ligand_input_sdf": "/l.sdf"}
    assert _resolve_paths(row) == ("/r.pdb", "/l.sdf")


def test_resolve_paths_benchmark_columns():
    row = {"protein_path": "/p.pdb", "ligand_path": "/l.sdf"}
    assert _resolve_paths(row) == ("/p.pdb", "/l.sdf")


def test_cache_key_stable():
    a = Path("/foo/a.pdb")
    b = Path("/bar/l.sdf")
    k1 = _cache_key(a, b)
    k2 = _cache_key(a, b)
    assert k1 == k2
    assert len(k1) == 64
