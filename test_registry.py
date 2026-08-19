"""
Unit tests for src/registry.py — the pure data-layer CRUD for the person registry (.npz files).
No Tkinter/GUI involved. Each test points registry.REGISTRY_DIR at a fresh temp directory so
tests never touch the real logs/registry/.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from src import registry


class RegistrySanitizeTestCase(unittest.TestCase):
    def test_spaces_become_underscore(self):
        self.assertEqual(registry.sanitize_person_name("Nguyen Van A"), "Nguyen_Van_A")

    def test_special_characters_stripped(self):
        self.assertEqual(registry.sanitize_person_name("An@#$%^&*()!"), "An")

    def test_diacritics_preserved(self):
        self.assertEqual(registry.sanitize_person_name("Nguyễn Đức"), "Nguyễn_Đức")

    def test_path_traversal_neutralized(self):
        result = registry.sanitize_person_name("../../etc/passwd")
        self.assertNotIn("..", result)
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            registry.sanitize_person_name("")

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            registry.sanitize_person_name(None)

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            registry.sanitize_person_name("   ")

    def test_only_special_characters_raises(self):
        with self.assertRaises(ValueError):
            registry.sanitize_person_name("###$$$!!!")

    def test_repeated_underscores_collapsed(self):
        self.assertEqual(registry.sanitize_person_name("A   B"), "A_B")


class RegistryCrudTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="registry_test_")
        self._orig_dir = registry.REGISTRY_DIR
        registry.REGISTRY_DIR = self.tmpdir

    def tearDown(self):
        registry.REGISTRY_DIR = self._orig_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_multiple_people_stay_isolated(self):
        emb_a = np.random.rand(512).astype(np.float32)
        emb_b = np.random.rand(512).astype(np.float32)

        path_a = registry.save_person("Alice", emb_a, aspect_ratio=0.5, sample_count=10)
        path_b = registry.save_person("Bob", emb_b, aspect_ratio=0.6, sample_count=15)

        self.assertNotEqual(path_a, path_b)
        self.assertTrue(os.path.exists(path_a))
        self.assertTrue(os.path.exists(path_b))

        loaded_a = registry.load_person(path_a)
        loaded_b = registry.load_person(path_b)
        np.testing.assert_array_almost_equal(loaded_a["embedding"], emb_a)
        np.testing.assert_array_almost_equal(loaded_b["embedding"], emb_b)
        self.assertAlmostEqual(loaded_a["aspect_ratio"], 0.5)
        self.assertAlmostEqual(loaded_b["aspect_ratio"], 0.6)
        self.assertEqual(loaded_a["sample_count"], 10)
        self.assertEqual(loaded_b["sample_count"], 15)

        entries = {e["name"]: e for e in registry.list_registry()}
        self.assertEqual(set(entries.keys()), {"Alice", "Bob"})

    def test_save_overwrites_same_name_only(self):
        registry.save_person("Alice", np.zeros(4, dtype=np.float32), 0.5, 5)
        registry.save_person("Bob", np.ones(4, dtype=np.float32), 0.5, 5)

        new_emb = np.full(4, 9.0, dtype=np.float32)
        registry.save_person("Alice", new_emb, 0.7, 20)

        entries = registry.list_registry()
        self.assertEqual(len(entries), 2)  # still exactly 2 people, not 3
        alice = registry.load_person(registry.registry_path("Alice"))
        bob = registry.load_person(registry.registry_path("Bob"))
        np.testing.assert_array_almost_equal(alice["embedding"], new_emb)
        self.assertEqual(alice["sample_count"], 20)
        np.testing.assert_array_almost_equal(bob["embedding"], np.ones(4, dtype=np.float32))

    def test_delete_removes_only_target(self):
        registry.save_person("Alice", np.zeros(4, dtype=np.float32), 0.5, 5)
        registry.save_person("Bob", np.ones(4, dtype=np.float32), 0.5, 5)

        registry.delete_person("Alice")

        self.assertFalse(os.path.exists(registry.registry_path("Alice")))
        self.assertTrue(os.path.exists(registry.registry_path("Bob")))
        names = {e["name"] for e in registry.list_registry()}
        self.assertEqual(names, {"Bob"})

    def test_delete_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            registry.delete_person("Nobody")

    def test_rename_moves_file_and_preserves_data(self):
        emb = np.random.rand(4).astype(np.float32)
        registry.save_person("Alice", emb, aspect_ratio=0.55, sample_count=8)

        new_path = registry.rename_person("Alice", "Alicia")

        self.assertFalse(os.path.exists(registry.registry_path("Alice")))
        self.assertTrue(os.path.exists(new_path))
        loaded = registry.load_person(new_path)
        np.testing.assert_array_almost_equal(loaded["embedding"], emb)
        self.assertAlmostEqual(loaded["aspect_ratio"], 0.55)
        self.assertEqual(loaded["sample_count"], 8)

    def test_rename_to_existing_name_raises_without_deleting_original(self):
        registry.save_person("Alice", np.zeros(4, dtype=np.float32), 0.5, 5)
        registry.save_person("Bob", np.ones(4, dtype=np.float32), 0.5, 5)

        with self.assertRaises(FileExistsError):
            registry.rename_person("Alice", "Bob")

        # Neither original file should have been touched by the failed rename.
        self.assertTrue(os.path.exists(registry.registry_path("Alice")))
        self.assertTrue(os.path.exists(registry.registry_path("Bob")))

    def test_rename_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            registry.rename_person("Nobody", "SomeoneElse")

    def test_empty_registry_returns_empty_list(self):
        self.assertEqual(registry.list_registry(), [])

    def test_person_exists(self):
        self.assertFalse(registry.person_exists("Alice"))
        registry.save_person("Alice", np.zeros(4, dtype=np.float32), 0.5, 5)
        self.assertTrue(registry.person_exists("Alice"))

    def test_list_registry_skips_unreadable_file(self):
        registry.save_person("Alice", np.zeros(4, dtype=np.float32), 0.5, 5)
        corrupt_path = os.path.join(self.tmpdir, "Corrupt.npz")
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("not a real npz file")

        entries = registry.list_registry()
        names = {e["name"] for e in entries}
        self.assertEqual(names, {"Alice"})  # corrupt entry skipped, not raised


if __name__ == "__main__":
    unittest.main()
