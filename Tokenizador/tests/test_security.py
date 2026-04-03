from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from security_utils import load_dataset_secure, maybe_login_from_env, validate_uint16_ids


class SecurityUtilsTests(unittest.TestCase):
    def test_remote_code_is_blocked_by_default(self):
        with self.assertRaises(RuntimeError):
            load_dataset_secure("dummy/dataset", split="train", streaming=True, trust_remote_code=True)

    def test_login_is_optional_without_token(self):
        self.assertFalse(maybe_login_from_env())

    def test_uint16_validation_rejects_out_of_range_ids(self):
        with self.assertRaises(ValueError):
            validate_uint16_ids([0, 1, 70000], source_name="smoke")


if __name__ == "__main__":
    unittest.main()
