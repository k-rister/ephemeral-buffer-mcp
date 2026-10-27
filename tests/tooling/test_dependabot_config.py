"""Tests for Dependabot update coverage."""

import unittest

import yaml

from tests.paths import REPOSITORY_ROOT


DEPENDABOT_CONFIG = REPOSITORY_ROOT / ".github" / "dependabot.yml"


class TestDependabotConfig(unittest.TestCase):
    def test_pip_updates_cover_declared_dependencies_weekly(self):
        config = yaml.safe_load(DEPENDABOT_CONFIG.read_text(encoding="utf-8"))
        pip_updates = [
            update
            for update in config["updates"]
            if update["package-ecosystem"] == "pip"
        ]

        self.assertEqual(len(pip_updates), 1)
        pip_update = pip_updates[0]
        self.assertEqual(pip_update["directory"], "/")
        self.assertEqual(pip_update["schedule"]["interval"], "weekly")
        self.assertNotIn("allow", pip_update)


if __name__ == "__main__":
    unittest.main()
