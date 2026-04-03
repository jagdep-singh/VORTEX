from __future__ import annotations

import unittest
from unittest.mock import patch

from click.testing import CliRunner

import main


class MainCliTests(unittest.TestCase):
    def test_update_flag_runs_update_flow_before_startup(self) -> None:
        runner = CliRunner()

        with patch("main._run_update_flow", return_value=0) as run_update:
            with patch("main._choose_startup_workspace") as choose_workspace:
                result = runner.invoke(main.main, ["--update"])

        self.assertEqual(result.exit_code, 0)
        run_update.assert_called_once_with()
        choose_workspace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
