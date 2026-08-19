"""
Unit tests for src/person_selector.py's PersonRegistrySelector.run() ORCHESTRATION logic
(empty-registry auto-redirect, close-without-selecting). These deliberately do NOT open a real
Tk window: _register_new_blocking() and _show_list_and_wait() are mocked out, since they're the
only methods that touch Tkinter — everything else in run() is plain control flow and is exactly
what's under test here.
"""
import unittest
from unittest.mock import patch, MagicMock

from src.person_selector import PersonRegistrySelector


class PersonRegistrySelectorTestCase(unittest.TestCase):
    def _make_selector(self):
        return PersonRegistrySelector(config={}, config_path="unused.yaml")

    @patch("src.person_selector.registry.list_registry")
    def test_empty_registry_auto_redirects_to_registration(self, mock_list_registry):
        # First call (before registering): empty. Second call (after auto-registration): 1 entry.
        mock_list_registry.side_effect = [
            [],
            [{"name": "Alice", "path": "logs/registry/Alice.npz", "created_at": "t", "sample_count": 5}],
        ]
        selector = self._make_selector()
        selector._register_new_blocking = MagicMock()
        selector._show_list_and_wait = MagicMock()

        selector.run()

        selector._register_new_blocking.assert_called_once()
        selector._show_list_and_wait.assert_called_once()
        # The list handed to the (mocked) list-building step must be the POST-registration one.
        shown_entries = selector._show_list_and_wait.call_args[0][0]
        self.assertEqual(shown_entries[0]["name"], "Alice")

    @patch("src.person_selector.registry.list_registry")
    def test_empty_registry_stays_empty_after_failed_registration_returns_none(self, mock_list_registry):
        # Registration attempt didn't produce anything (e.g. user cancelled or it failed).
        mock_list_registry.side_effect = [[], []]
        selector = self._make_selector()
        selector._register_new_blocking = MagicMock()
        selector._show_list_and_wait = MagicMock()

        result = selector.run()

        self.assertIsNone(result)
        selector._register_new_blocking.assert_called_once()
        selector._show_list_and_wait.assert_not_called()  # nothing to show, must not open the list GUI

    @patch("src.person_selector.registry.list_registry")
    def test_nonempty_registry_skips_auto_registration(self, mock_list_registry):
        entries = [{"name": "Bob", "path": "logs/registry/Bob.npz", "created_at": "t", "sample_count": 3}]
        mock_list_registry.return_value = entries
        selector = self._make_selector()
        selector._register_new_blocking = MagicMock()
        selector._show_list_and_wait = MagicMock()

        selector.run()

        selector._register_new_blocking.assert_not_called()
        selector._show_list_and_wait.assert_called_once_with(entries)

    @patch("src.person_selector.registry.list_registry")
    def test_closing_without_selecting_returns_none(self, mock_list_registry):
        entries = [{"name": "Bob", "path": "logs/registry/Bob.npz", "created_at": "t", "sample_count": 3}]
        mock_list_registry.return_value = entries
        selector = self._make_selector()

        # Simulate the user closing the window: _show_list_and_wait runs (would normally block
        # on mainloop and only return once the window closes) but never sets selected_path,
        # exactly like _on_close() leaves it at its initial None.
        def fake_show_and_wait(_entries):
            pass
        selector._show_list_and_wait = MagicMock(side_effect=fake_show_and_wait)

        result = selector.run()

        self.assertIsNone(result)

    @patch("src.person_selector.registry.registry_path")
    @patch("src.person_selector.registry.list_registry")
    def test_selecting_a_person_returns_their_path(self, mock_list_registry, mock_registry_path):
        entries = [{"name": "Bob", "path": "logs/registry/Bob.npz", "created_at": "t", "sample_count": 3}]
        mock_list_registry.return_value = entries
        mock_registry_path.return_value = "logs/registry/Bob.npz"
        selector = self._make_selector()

        # Simulate _on_select() being triggered by the (mocked-out) GUI.
        def fake_show_and_wait(_entries):
            selector.selected_path = mock_registry_path("Bob")
        selector._show_list_and_wait = MagicMock(side_effect=fake_show_and_wait)

        result = selector.run()

        self.assertEqual(result, "logs/registry/Bob.npz")


if __name__ == "__main__":
    unittest.main()
