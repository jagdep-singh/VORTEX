from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from utils.versioning import (
    VersionManager,
    is_newer_version,
    recommended_update_instruction,
)


class VersioningTests(unittest.TestCase):
    def test_is_newer_version_handles_basic_semver(self) -> None:
        self.assertTrue(is_newer_version("0.1.0.3", "0.1.0.2"))
        self.assertTrue(is_newer_version("1.0.0", "0.9.9"))
        self.assertFalse(is_newer_version("0.1.0.2", "0.1.0.2"))
        self.assertFalse(is_newer_version("0.1.0.1", "0.1.0.2"))

    def test_recommended_update_instruction_changes_for_local_source(self) -> None:
        self.assertEqual(recommended_update_instruction("pipx"), "run vortex --update")
        self.assertEqual(
            recommended_update_instruction("editable"),
            "pull the latest repo changes",
        )

    def test_get_release_info_uses_fresh_cache(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manager = VersionManager(
                data_dir=Path(tmpdir),
                current_version="0.1.0.2",
                install_mode="pipx",
            )

            with patch.object(manager, "fetch_latest_version", return_value="0.1.0.3"):
                live_info = manager.get_release_info(force_refresh=True)

            self.assertEqual(live_info.latest_version, "0.1.0.3")
            self.assertEqual(live_info.source, "live")

            with patch.object(manager, "fetch_latest_version") as fetch_latest_version:
                cached_info = manager.get_release_info()

            fetch_latest_version.assert_not_called()
            self.assertEqual(cached_info.latest_version, "0.1.0.3")
            self.assertEqual(cached_info.source, "cache")

    def test_perform_self_update_uses_pipx_when_available(self) -> None:
        manager = VersionManager(
            current_version="0.1.0.2",
            install_mode="pipx",
        )

        completed_process = Mock(returncode=0)
        with patch("utils.versioning.shutil.which", return_value="/usr/local/bin/pipx"):
            with patch(
                "utils.versioning.subprocess.run",
                return_value=completed_process,
            ) as run:
                result = manager.perform_self_update()

        run.assert_called_once_with(
            ["/usr/local/bin/pipx", "upgrade", "vortex-agent-cli"],
            check=False,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.command, "pipx upgrade vortex-agent-cli")

    def test_perform_self_update_skips_editable_install(self) -> None:
        manager = VersionManager(
            current_version="0.1.0.2",
            install_mode="editable",
        )

        with patch("utils.versioning.subprocess.run") as run:
            result = manager.perform_self_update()

        run.assert_not_called()
        self.assertTrue(result.success)
        self.assertIn("local checkout", result.message)


if __name__ == "__main__":
    unittest.main()
