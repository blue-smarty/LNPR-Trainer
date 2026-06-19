"""Tests for dataset path resolution utilities in dashboard/path_utils.py.

These tests verify that:
- Relative paths are always resolved against REPO_ROOT, never against CWD.
- Paths that already contain a ``dashboard/`` segment are NOT double-prefixed.
- Absolute paths are returned unchanged.
- Directory inputs resolve to ``data.yaml`` inside the directory.
- ZIP inputs are extracted and ``data.yaml`` is located inside.
- No duplicated path segment (``dashboard/dashboard``) appears in the result.
"""

from __future__ import annotations

import os
import sys
import unittest
import zipfile
import shutil
import tempfile
from pathlib import Path

# Ensure the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.path_utils import (  # noqa: E402
    resolve_dataset_path_or_url,
    resolve_local_path,
    REPO_ROOT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zip_with_yaml(dest: Path, yaml_content: str = "nc: 1\n") -> Path:
    """Create a ZIP archive at *dest* containing a ``data.yaml`` file."""
    zip_path = dest / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.yaml", yaml_content)
    return zip_path


def _make_dir_with_yaml(dest: Path, yaml_content: str = "nc: 1\n") -> Path:
    """Create a directory at *dest* containing a ``data.yaml`` file."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "data.yaml").write_text(yaml_content, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Tests for resolve_dataset_path_or_url
# ---------------------------------------------------------------------------


class TestResolveDatasetPathOrUrl(unittest.TestCase):
    """Unit tests for resolve_dataset_path_or_url."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="lnpr_test_")
        self._tmp_path = Path(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ── Directory input ───────────────────────────────────────────────────────

    def test_directory_resolves_to_data_yaml(self) -> None:
        """A directory path should resolve to the data.yaml inside it."""
        dataset_dir = self._tmp_path / "mydata"
        _make_dir_with_yaml(dataset_dir)

        result = resolve_dataset_path_or_url(str(dataset_dir), self._tmp_path / "tmp")
        self.assertEqual(result, dataset_dir / "data.yaml")

    def test_directory_missing_yaml_raises(self) -> None:
        """A directory without data.yaml should raise FileNotFoundError."""
        empty_dir = self._tmp_path / "emptydata"
        empty_dir.mkdir()

        with self.assertRaises(FileNotFoundError):
            resolve_dataset_path_or_url(str(empty_dir), self._tmp_path / "tmp")

    # ── ZIP input ─────────────────────────────────────────────────────────────

    def test_zip_resolves_to_data_yaml(self) -> None:
        """A .zip file input should extract and return the data.yaml path."""
        zip_path = _make_zip_with_yaml(self._tmp_path)
        extract_tmp = self._tmp_path / "extract_tmp"
        extract_tmp.mkdir()

        result = resolve_dataset_path_or_url(str(zip_path), extract_tmp)
        self.assertEqual(result.name, "data.yaml")
        self.assertTrue(result.exists())

    def test_zip_missing_yaml_raises(self) -> None:
        """A .zip without data.yaml should raise FileNotFoundError."""
        zip_path = self._tmp_path / "no_yaml.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("images/train/placeholder.txt", "")

        with self.assertRaises(FileNotFoundError):
            resolve_dataset_path_or_url(str(zip_path), self._tmp_path / "tmp2")

    # ── Absolute path input ───────────────────────────────────────────────────

    def test_absolute_yaml_path_returned_as_is(self) -> None:
        """An absolute path to data.yaml should be returned unchanged."""
        yaml_file = self._tmp_path / "data.yaml"
        yaml_file.write_text("nc: 1\n", encoding="utf-8")

        result = resolve_dataset_path_or_url(str(yaml_file), self._tmp_path / "tmp")
        self.assertEqual(result, yaml_file)

    # ── Relative path without dashboard prefix ────────────────────────────────

    def test_relative_path_without_dashboard_prefix(self) -> None:
        """Relative path not starting with dashboard/ resolves from REPO_ROOT."""
        dataset_dir = REPO_ROOT / "data" / "lnpr_dataset"
        yaml_file = dataset_dir / "data.yaml"
        created = False
        try:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            if not yaml_file.exists():
                yaml_file.write_text("nc: 1\n", encoding="utf-8")
                created = True

            result = resolve_dataset_path_or_url(
                "data/lnpr_dataset/data.yaml", self._tmp_path / "tmp"
            )
            self.assertEqual(result, yaml_file.resolve())
        finally:
            if created and yaml_file.exists():
                yaml_file.unlink()

    # ── Relative path that already includes dashboard/ ────────────────────────

    def test_relative_path_with_dashboard_prefix_no_duplication(self) -> None:
        """Relative path 'dashboard/...' must NOT produce 'dashboard/dashboard/'."""
        dataset_dir = REPO_ROOT / "dashboard" / "data" / "lnpr_dataset"
        yaml_file = dataset_dir / "data.yaml"
        created_dir = not dataset_dir.exists()
        created_file = False
        try:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            if not yaml_file.exists():
                yaml_file.write_text("nc: 1\n", encoding="utf-8")
                created_file = True

            result = resolve_dataset_path_or_url(
                "dashboard/data/lnpr_dataset/data.yaml", self._tmp_path / "tmp"
            )
            result_str = str(result)
            self.assertNotIn("dashboard/dashboard", result_str,
                             "Duplicated 'dashboard' segment found in resolved path")
            self.assertEqual(result, yaml_file.resolve())
        finally:
            if created_file and yaml_file.exists():
                yaml_file.unlink()
            if created_dir and dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)

    # ── Non-existent path ─────────────────────────────────────────────────────

    def test_nonexistent_path_raises(self) -> None:
        """A path that doesn't exist should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            resolve_dataset_path_or_url(
                "/nonexistent/path/data.yaml", self._tmp_path / "tmp"
            )


# ---------------------------------------------------------------------------
# Tests for resolve_local_path  (the fix used in the selectbox fallback branch)
# ---------------------------------------------------------------------------


class TestResolveLocalPath(unittest.TestCase):
    """Tests for resolve_local_path — the canonical path normaliser.

    These replicate the 'else' branch fix in app.py where the selected
    data.yaml relative path must be anchored to REPO_ROOT, not to CWD.
    """

    def test_relative_path_resolves_against_repo_root(self) -> None:
        """Relative path resolves under REPO_ROOT, not the process CWD."""
        result = resolve_local_path("data/lnpr_dataset/data.yaml")
        expected = (REPO_ROOT / "data/lnpr_dataset/data.yaml").resolve()
        self.assertEqual(result, expected)

    def test_relative_dashboard_prefix_no_duplication(self) -> None:
        """'dashboard/...' relative path never doubles to 'dashboard/dashboard/...'."""
        result = resolve_local_path("dashboard/data/lnpr_dataset/data.yaml")
        result_str = str(result)
        self.assertNotIn("dashboard/dashboard", result_str,
                         "Duplicated 'dashboard' segment found in resolved path")
        expected = (REPO_ROOT / "dashboard/data/lnpr_dataset/data.yaml").resolve()
        self.assertEqual(result, expected)

    def test_absolute_path_unchanged(self) -> None:
        """An absolute path is returned as-is after resolve()."""
        abs_path = "/tmp/myproject/data/lnpr_dataset/data.yaml"
        result = resolve_local_path(abs_path)
        self.assertEqual(result, Path(abs_path).resolve())

    def test_result_is_always_absolute(self) -> None:
        """The resolved path is always absolute."""
        for rel in ("data/lnpr_dataset/data.yaml", "dashboard/data/lnpr_dataset/data.yaml"):
            with self.subTest(rel=rel):
                result = resolve_local_path(rel)
                self.assertTrue(result.is_absolute(),
                                f"Expected absolute path, got: {result}")

    def test_cwd_does_not_affect_result(self) -> None:
        """Changing CWD must not change the resolved path.

        This is the direct regression test for the original bug:
        when Streamlit was launched from ``dashboard/``, the old code produced
        ``…/dashboard/dashboard/data/lnpr_dataset/data.yaml`` because it used
        ``Path(data_yaml).resolve()`` (CWD-relative) instead of anchoring to
        REPO_ROOT.
        """
        data_yaml = "dashboard/data/lnpr_dataset/data.yaml"
        result_before = resolve_local_path(data_yaml)

        orig_cwd = os.getcwd()
        try:
            os.chdir(REPO_ROOT / "dashboard")
            result_after = resolve_local_path(data_yaml)
        finally:
            os.chdir(orig_cwd)

        self.assertEqual(result_before, result_after,
                         "Path resolution changed when CWD switched to dashboard/")

    def test_custom_base(self) -> None:
        """resolve_local_path respects a custom base directory."""
        custom_base = Path("/some/custom/base")
        result = resolve_local_path("data/lnpr_dataset/data.yaml", base=custom_base)
        expected = (custom_base / "data/lnpr_dataset/data.yaml").resolve()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
