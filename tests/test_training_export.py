import unittest

from training_export import (
    has_independent_oracle,
    render_training_examples,
    select_eval_episodes,
)


def curated_example() -> dict:
    return {
        "schema_version": 1,
        "episode_id": "episode",
        "context_id": "harmful",
        "view": "minimal_user",
        "messages": [{"role": "user", "content": "unauthorized"}],
        "command": "seed",
        "scenario_harm_label": "harmful",
        "decision_label": "deny",
    }


def attempt(
    *,
    command: str,
    verdict: str = "PRESERVED",
    confidence: float = 0.9,
    oracle_model: str = "oracle",
) -> dict:
    return {
        "run_id": "run",
        "attempt_index": 2,
        "episode_id": "episode",
        "context_id": "harmful",
        "classifier_view": "minimal_user",
        "classifier_messages": [{"role": "user", "content": "unauthorized"}],
        "command": command,
        "target": {"decision": "BENIGN"},
        "oracle": {"verdict": verdict, "confidence": confidence},
        "models": {
            "target": {
                "provider": "bedrock",
                "model_id": "target",
                "base_url": None,
            },
            "oracle": {
                "provider": "bedrock",
                "model_id": oracle_model,
                "base_url": None,
            },
        },
    }


class TrainingExportTest(unittest.TestCase):
    def test_includes_only_independently_oracle_preserved_attempts(self) -> None:
        examples, stats = render_training_examples(
            [curated_example()],
            [
                attempt(command="variant"),
                attempt(command="lost-effect", verdict="NOT_PRESERVED"),
                attempt(command="correlated", oracle_model="target"),
            ],
            oracle_min_confidence=0.75,
            allow_correlated_oracle=False,
        )

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[-1]["command"], "variant")
        self.assertEqual(examples[-1]["decision_label"], "deny")
        self.assertEqual(stats["oracle_rejected"], 1)
        self.assertEqual(stats["correlated_oracle_rejected"], 1)

    def test_detects_oracle_independence_by_provider_and_model(self) -> None:
        self.assertTrue(has_independent_oracle(attempt(command="variant")))
        self.assertFalse(
            has_independent_oracle(
                attempt(command="variant", oracle_model="target")
            )
        )

    def test_split_is_stable_and_grouped_by_episode(self) -> None:
        examples = [
            curated_example(),
            {
                **curated_example(),
                "episode_id": "second",
                "decision_label": "allow",
            },
        ]
        selected = select_eval_episodes(examples, 0.5)
        self.assertEqual(selected, select_eval_episodes(examples, 0.5))
        self.assertEqual(len(selected), 1)
        with self.assertRaisesRegex(ValueError, "eval_fraction"):
            select_eval_episodes(examples, 1.0)


if __name__ == "__main__":
    unittest.main()
