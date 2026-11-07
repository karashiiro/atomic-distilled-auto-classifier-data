import tempfile
import unittest
from pathlib import Path

from poc_pipeline import ROOT, load_yaml
from scale_inputs import write_yaml
from workflow import (
    batch_promote,
    split_tasks_by_episode,
    write_review_queue,
)


class WorkflowTest(unittest.TestCase):
    def test_tasks_are_partitioned_by_episode(self) -> None:
        curated = [
            {
                "episode_id": episode_id,
                "decision_label": label,
            }
            for episode_id, label in (
                ("one", "deny"),
                ("one", "allow"),
                ("two", "deny"),
                ("three", "request_context"),
            )
        ]
        tasks = [
            {"episode_id": episode_id}
            for episode_id in ("one", "two", "three")
        ]

        training, evaluation, eval_ids = split_tasks_by_episode(
            tasks,
            curated,
            0.34,
        )

        self.assertTrue(eval_ids)
        self.assertFalse(
            {task["episode_id"] for task in training}
            & {task["episode_id"] for task in evaluation}
        )

    def test_review_queue_contains_resolved_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft = load_yaml(
                ROOT / "poc_data" / "reviews" / "cron_replace_crontab.yaml"
            )
            draft["review"] = {
                "suitability": "accepted",
                "automated_review": {
                    "verdict": "approved",
                    "confidence": 1.0,
                },
            }
            write_yaml(root / "drafts" / "cron.yaml", draft)
            output = root / "queue.md"

            counts = write_review_queue(
                root / "drafts",
                ROOT / "poc_data" / "atomic",
                output,
            )

            self.assertEqual(counts["accepted"], 1)
            self.assertIn("crontab", output.read_text(encoding="utf-8"))

    def test_batch_promotion_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft = load_yaml(
                ROOT / "poc_data" / "reviews" / "cron_replace_crontab.yaml"
            )
            draft["review"] = {
                "status": "draft",
                "reviewer": "",
                "suitability": "accepted",
                "automated_review": {
                    "verdict": "approved",
                    "confidence": 0.99,
                },
            }
            write_yaml(root / "drafts" / "cron.yaml", draft)
            arguments = {
                "drafts_dir": root / "drafts",
                "reviews_dir": root / "reviews",
                "atomic_dir": ROOT / "poc_data" / "atomic",
                "pinned_atomic_dir": root / "atomic",
                "reviewer": "reviewer",
                "min_confidence": 0.8,
            }

            first = batch_promote(**arguments)
            second = batch_promote(**arguments)

            self.assertEqual(first["promoted"], 1)
            self.assertEqual(second["already_promoted"], 1)


if __name__ == "__main__":
    unittest.main()
