import json
import tempfile
import unittest
from pathlib import Path

from poc_pipeline import (
    ROOT,
    build_episodes,
    build_outputs,
    load_atomic_tests,
    render_classifier_examples,
    render_generator_tasks,
    validate_visible_input_labels,
)


class PocPipelineTest(unittest.TestCase):
    def test_atomic_loader_ignores_non_technique_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            atomic_dir = Path(temporary)
            (atomic_dir / "index.yaml").write_text(
                "name: supporting index\n",
                encoding="utf-8",
            )
            self.assertEqual(load_atomic_tests(atomic_dir), {})

    def setUp(self) -> None:
        self.episodes = build_episodes(
            ROOT / "poc_data" / "atomic",
            ROOT / "poc_data" / "reviews",
        )

    def test_builds_reviewed_episode(self) -> None:
        self.assertGreaterEqual(len(self.episodes), 5)
        episode = next(
            episode
            for episode in self.episodes
            if episode["episode_id"] == "cron-replace-crontab-v1"
        )
        self.assertEqual(episode["source"]["technique_id"], "T1053.003")
        self.assertNotIn("#{", episode["execution"]["command"])
        self.assertNotIn("evil", episode["execution"]["command"])
        self.assertIn("/opt/acme/health-check.sh", episode["execution"]["command"])

    def test_renders_context_ablation_views(self) -> None:
        examples = render_classifier_examples(self.episodes)
        expected_count = sum(
            len(episode["contexts"]) * 4
            for episode in self.episodes
        )
        self.assertEqual(len(examples), expected_count)
        self.assertGreaterEqual(
            len({example["command"] for example in examples}),
            5,
        )
        self.assertTrue(all("source" not in example for example in examples))
        self.assertTrue(all("behavior" not in example for example in examples))

        decisions = {
            (example["context_id"], example["view"]): example["decision_label"]
            for example in examples
            if example["episode_id"] == "cron-replace-crontab-v1"
        }
        self.assertEqual(
            decisions[("unauthorized_customer_host", "all_user")],
            "deny",
        )
        self.assertEqual(
            decisions[("authorized_admin", "all_user")],
            "allow",
        )
        self.assertEqual(
            decisions[("authorized_admin", "last_user")],
            "request_context",
        )

    def test_rejects_conflicting_labels_for_identical_visible_inputs(self) -> None:
        examples = [
            {
                "messages": [],
                "command": "example",
                "decision_label": "allow",
            },
            {
                "messages": [],
                "command": "example",
                "decision_label": "deny",
            },
        ]
        with self.assertRaisesRegex(ValueError, "contradictory"):
            validate_visible_input_labels(examples)

    def test_generator_only_receives_harmful_contexts(self) -> None:
        tasks = render_generator_tasks(self.episodes)
        expected_count = sum(
            context["harm_label"] == "harmful"
            for episode in self.episodes
            for context in episode["contexts"]
        )
        self.assertEqual(len(tasks), expected_count)
        self.assertTrue(
            all(task["context_id"].startswith("unauthorized") for task in tasks)
        )
        for task in tasks:
            self.assertTrue(task["behavior"]["required_effects"])
            self.assertEqual(set(task["classifier_views"]), {
                "command_only",
                "last_user",
                "minimal_user",
                "all_user",
            })
            self.assertTrue(task["oracle_context"])

    def test_writes_valid_jsonl_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            counts = build_outputs(
                ROOT / "poc_data" / "atomic",
                ROOT / "poc_data" / "reviews",
                output_dir,
            )
            self.assertEqual(
                counts,
                (
                    len(self.episodes),
                    len(render_classifier_examples(self.episodes)),
                    len(render_generator_tasks(self.episodes)),
                ),
            )

            for filename in (
                "episodes.jsonl",
                "classifier_examples.jsonl",
                "generator_tasks.jsonl",
            ):
                lines = (output_dir / filename).read_text().splitlines()
                self.assertTrue(lines)
                for line in lines:
                    self.assertIsInstance(json.loads(line), dict)


if __name__ == "__main__":
    unittest.main()
