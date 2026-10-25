import json
import os
import tempfile
import unittest

from caisen.store import connect, get_pattern, save_pattern, save_patterns


class StoreTests(unittest.TestCase):
    def test_batch_saves_patterns_and_snapshots_in_one_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                save_patterns([
                    {"pattern_id": "a", "data_version": "v1", "state": "forming"},
                    {"pattern_id": "b", "data_version": "v1", "state": "confirmed"},
                ])
                self.assertEqual(get_pattern("a")["state"], "forming")
                with connect() as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM pattern_snapshots").fetchone()[0], 2)
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_data_versions_are_kept_as_immutable_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                base = {"pattern_id":"abc", "data_version":"cutoff-a", "state":"confirmed"}
                save_pattern(base)
                newer = dict(base, data_version="cutoff-b", state="invalidated")
                save_pattern(newer)
                with connect() as db:
                    rows=db.execute("SELECT data_version,payload FROM pattern_snapshots ORDER BY data_version").fetchall()
                self.assertEqual([r[0] for r in rows], ["cutoff-a","cutoff-b"])
                self.assertEqual(json.loads(rows[0][1])["state"],"confirmed")
                self.assertEqual(get_pattern("abc")["state"],"invalidated")
            finally:
                if previous is None: os.environ.pop("CAISEN_DB",None)
                else: os.environ["CAISEN_DB"] = previous


if __name__ == "__main__": unittest.main()
