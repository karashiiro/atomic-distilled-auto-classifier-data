import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from poc_pipeline import ROOT, validate_visible_input_labels, write_jsonl

DEFAULT_CLASSIFIER_PATH = ROOT / "generated" / "classifier_examples.jsonl"
DEFAULT_ATTEMPTS_PATH = ROOT / "adversarial_attempts.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "training_examples.jsonl"


def load_jsonl(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if missing_ok and not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def has_independent_oracle(record: dict[str, Any]) -> bool:
    models = record["models"]
    target = models["target"]
    oracle = models["oracle"]
    return (
        target.get("provider", "openai"),
        target["model_id"],
    ) != (
        oracle.get("provider", "openai"),
        oracle["model_id"],
    )


def select_eval_episodes(
    curated: list[dict[str, Any]],
    eval_fraction: float,
) -> set[str]:
    if not 0 <= eval_fraction < 1:
        raise ValueError("eval_fraction must be at least 0 and less than 1")
    labels_by_episode: dict[str, set[str]] = {}
    for example in curated:
        labels_by_episode.setdefault(example["episode_id"], set()).add(
            example["decision_label"]
        )
    if eval_fraction == 0 or len(labels_by_episode) < 2:
        return set()

    eval_count = max(1, round(len(labels_by_episode) * eval_fraction))
    eval_count = min(eval_count, len(labels_by_episode) - 1)
    ranked = sorted(
        labels_by_episode,
        key=lambda episode_id: (
            -len(labels_by_episode[episode_id]),
            hashlib.sha256(episode_id.encode("utf-8")).hexdigest(),
        ),
    )
    return set(ranked[:eval_count])


def render_training_examples(
    curated: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    *,
    oracle_min_confidence: float,
    allow_correlated_oracle: bool,
    eval_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eval_episodes = select_eval_episodes(curated, eval_fraction)
    examples = [
        {
            **example,
            "split": (
                "eval"
                if example["episode_id"] in eval_episodes
                else "train"
            ),
            "source": {
                "kind": "reviewed_atomic_context",
            },
        }
        for example in curated
    ]
    stats = {
        "curated": len(curated),
        "adversarial_included": 0,
        "oracle_rejected": 0,
        "correlated_oracle_rejected": 0,
        "unlabeled_rejected": 0,
        "duplicates": 0,
    }
    reviewed_labels = {
        (
            example["episode_id"],
            example["context_id"],
            example["view"],
        ): example["decision_label"]
        for example in curated
    }

    for attempt in attempts:
        oracle = attempt["oracle"]
        if (
            oracle["verdict"] != "PRESERVED"
            or oracle["confidence"] < oracle_min_confidence
        ):
            stats["oracle_rejected"] += 1
            continue
        if not allow_correlated_oracle and not has_independent_oracle(attempt):
            stats["correlated_oracle_rejected"] += 1
            continue
        expected_decision = attempt.get("expected_decision") or reviewed_labels.get(
            (
                attempt["episode_id"],
                attempt["context_id"],
                attempt["classifier_view"],
            )
        )
        if expected_decision is None:
            stats["unlabeled_rejected"] += 1
            continue

        examples.append(
            {
                "schema_version": 1,
                "episode_id": attempt["episode_id"],
                "context_id": attempt["context_id"],
                "view": attempt["classifier_view"],
                "messages": attempt["classifier_messages"],
                "command": attempt["command"],
                "scenario_harm_label": "harmful",
                "decision_label": expected_decision,
                "split": (
                    "eval"
                    if attempt["episode_id"] in eval_episodes
                    else "train"
                ),
                "source": {
                    "kind": "oracle_preserved_adversarial_attempt",
                    "run_id": attempt["run_id"],
                    "attempt_index": attempt["attempt_index"],
                    "trajectory_index": attempt.get("trajectory_index"),
                    "generation_mode": (
                        attempt.get("generation") or {}
                    ).get("mode"),
                    "target_decision": attempt["target"]["decision"],
                    "oracle_confidence": oracle["confidence"],
                },
            }
        )
        stats["adversarial_included"] += 1

    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example in examples:
        key = json.dumps(
            {
                "messages": example["messages"],
                "command": example["command"],
                "decision_label": example["decision_label"],
            },
            sort_keys=True,
        )
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        deduplicated.append(example)

    validate_visible_input_labels(deduplicated)
    return deduplicated, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export reviewed and oracle-approved classifier training data."
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=DEFAULT_CLASSIFIER_PATH,
    )
    parser.add_argument(
        "--attempts-path",
        type=Path,
        default=DEFAULT_ATTEMPTS_PATH,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--oracle-min-confidence", type=float, default=0.75)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument(
        "--allow-correlated-oracle",
        action="store_true",
        help="Include attempts where target and oracle use the same provider and model.",
    )
    args = parser.parse_args()

    examples, stats = render_training_examples(
        load_jsonl(args.classifier_path),
        load_jsonl(args.attempts_path, missing_ok=True),
        oracle_min_confidence=args.oracle_min_confidence,
        allow_correlated_oracle=args.allow_correlated_oracle,
        eval_fraction=args.eval_fraction,
    )
    write_jsonl(args.output_path, examples)
    print(
        f"Wrote {len(examples)} training examples "
        f"({stats['curated']} curated, "
        f"{stats['adversarial_included']} adversarial; "
        f"{stats['correlated_oracle_rejected']} correlated-oracle rejected, "
        f"{stats['oracle_rejected']} oracle rejected, "
        f"{stats['unlabeled_rejected']} unlabeled rejected, "
        f"{stats['duplicates']} duplicates removed)."
    )


if __name__ == "__main__":
    main()
