import io
import pickle
from pathlib import Path
import unittest

from tools.install_splits import RESOURCES, SplitUnpickler, read_records, relative_path, verified_records


class SplitTests(unittest.TestCase):
    def test_all_released_splits_match_hashes_and_csv(self):
        for seed in (1, 2, 3):
            with self.subTest(seed=seed):
                self.assertEqual(len(verified_records(seed)), 16000)

    def test_public_pickles_contain_only_relative_paths(self):
        for seed in (1, 2, 3):
            path = RESOURCES / "splits" / f"shot_16-seed_{seed}.pkl"
            with self.subTest(seed=seed):
                self.assertEqual(len(read_records(path, require_relative=True)), 16000)
                with path.open("rb") as stream:
                    items = SplitUnpickler(stream).load()["train"]
                self.assertTrue(all(item._impath.startswith("train/") for item in items))
                self.assertNotIn(b"/home/", path.read_bytes())
                self.assertNotIn(b"/data/", path.read_bytes())

    def test_paths_are_rebased_without_resampling(self):
        relative = "train/n00000001/n00000001_42.JPEG"
        self.assertEqual(relative_path(relative), relative)
        self.assertEqual(relative_path("/original/data/imagenet/" + relative), relative)

    def test_invalid_paths_are_rejected(self):
        for value in ["../image.jpg", "/etc/passwd", "train/../secret", "val/class/image.jpg", 123]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative_path(value)

    def test_unexpected_pickle_classes_are_rejected(self):
        serialized = pickle.dumps(Path("unrelated"))
        with self.assertRaises(pickle.UnpicklingError):
            SplitUnpickler(io.BytesIO(serialized)).load()


if __name__ == "__main__":
    unittest.main()
