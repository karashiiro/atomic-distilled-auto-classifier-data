import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adversarial_pipeline import (
    ModelSettings,
    load_tasks,
    run_adversarial_task,
)
from poc_pipeline import ROOT, build_outputs, load_atomic_tests, load_yaml
from scale_inputs import (
    ATOMIC_REVISION,
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    DEFAULT_CATALOG_PATH,
    DEFAULT_CHECKOUT,
    DEFAULT_DISTILLER_MODEL,
    DEFAULT_DRAFTS_DIR,
    DEFAULT_PINNED_ATOMIC_DIR,
    DEFAULT_REVIEWER_MODEL,
    DEFAULT_REVIEWER_FALLBACK_MODEL,
    DEFAULT_REVIEWS_DIR,
    catalog_candidates,
    draft_candidates,
    promote_draft,
    resolve_default_command,
    sync_atomic,
    write_jsonl,
    write_yaml,
)
from training_export import (
    load_jsonl,
    render_training_examples,
    select_eval_episodes,
)

GENERATED_DIR = ROOT / "generated"
REVIEW_QUEUE_PATH = GENERATED_DIR / "review_queue.md"
ATTEMPTS_PATH = GENERATED_DIR / "benchmark_attempts.jsonl"
BENCHMARK_ERRORS_PATH = GENERATED_DIR / "benchmark_errors.jsonl"
TRAINING_PATH = GENERATED_DIR / "training_examples.jsonl"
BENCHMARK_PATH = GENERATED_DIR / "benchmark_examples.jsonl"
MANIFEST_PATH = GENERATED_DIR / "benchmark_manifest.json"
AUGMENT_ATTEMPTS_PATH = GENERATED_DIR / "augmentation_attempts.jsonl"
AUGMENT_ERRORS_PATH = GENERATED_DIR / "augmentation_errors.jsonl"
AUGMENT_EXAMPLES_PATH = GENERATED_DIR / "augmentation_examples.jsonl"
AUGMENTED_TRAIN_PATH = GENERATED_DIR / "augmented_training_examples.jsonl"
AUGMENT_MANIFEST_PATH = GENERATED_DIR / "augmentation_manifest.json"


def split_tasks_by_episode(
    tasks: list[dict[str, Any]],
    curated: list[dict[str, Any]],
    eval_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    eval_episodes = select_eval_episodes(curated, eval_fraction)
    training_tasks = [
        task
        for task in tasks
        if task["episode_id"] not in eval_episodes
    ]
    eval_tasks = [
        task
        for task in tasks
        if task["episode_id"] in eval_episodes
    ]
    return training_tasks, eval_tasks, eval_episodes


def write_review_queue(
    drafts_dir: Path,
    atomic_dir: Path,
    output_path: Path,
) -> dict[str, int]:
    tests = load_atomic_tests(atomic_dir)
    sections = [
        "# Atomic Review Queue",
        "",
        "Generated drafts remain untrusted until batch promotion is explicitly confirmed.",
        "",
    ]
    counts = {"accepted": 0, "rejected": 0}
    for path in sorted(drafts_dir.glob("*.yaml")):
        draft = load_yaml(path)
        review = draft.get("review", {})
        suitability = review.get("suitability", "rejected")
        counts[suitability] = counts.get(suitability, 0) + 1
        guid = draft["source"]["atomic_guid"]
        test = tests.get(guid)
        command = resolve_default_command(test) if test is not None else None
        automated = review.get("automated_review") or {}
        sections.extend(
            [
                f"## {draft['id']}",
                "",
                f"- File: `{path}`",
                f"- Technique: `{draft['source']['technique_id']}`",
                f"- Atomic GUID: `{guid}`",
                f"- Automated suitability: `{suitability}`",
                (
                    "- Automated review: "
                    f"`{automated.get('verdict', 'not-run')}` "
                    f"({automated.get('confidence', 0):.2f})"
                ),
                (
                    "- Automated reviewer model: "
                    f"`{review.get('automated_review_model') or 'not-recorded'}`"
                ),
                f"- Objective: {draft['behavior']['objective']}",
                "",
                "```sh",
                command or "<unresolved command>",
                "```",
                "",
            ]
        )
        if review.get("rejection_reason"):
            sections.extend(
                [
                    f"Rejection reason: {review['rejection_reason']}",
                    "",
                ]
            )
        for context in draft.get("contexts", []):
            sections.extend(
                [
                    f"### Context: {context['id']} ({context['harm_label']})",
                    "",
                    *[
                        f"- {message}"
                        for message in context["user_messages"]
                    ],
                    "",
                    (
                        "Labels: "
                        f"`{json.dumps(context['view_labels'], sort_keys=True)}`"
                    ),
                    "",
                ]
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    return counts


def batch_promote(
    *,
    drafts_dir: Path,
    reviews_dir: Path,
    atomic_dir: Path,
    pinned_atomic_dir: Path,
    reviewer: str,
    min_confidence: float,
) -> dict[str, int]:
    counts = {
        "promoted": 0,
        "already_promoted": 0,
        "ineligible": 0,
    }
    for draft_path in sorted(drafts_dir.glob("*.yaml")):
        draft = load_yaml(draft_path)
        review = draft.get("review", {})
        automated = review.get("automated_review") or {}
        eligible = (
            review.get("suitability") == "accepted"
            and automated.get("verdict") == "approved"
            and automated.get("confidence", 0) >= min_confidence
        )
        if not eligible:
            counts["ineligible"] += 1
            continue

        output_path = reviews_dir / draft_path.name
        if output_path.exists():
            existing = load_yaml(output_path)
            if (
                existing["source"]["atomic_guid"]
                != draft["source"]["atomic_guid"]
            ):
                raise ValueError(f"conflicting promoted review: {output_path}")
            counts["already_promoted"] += 1
            continue

        review["status"] = "approved"
        review["reviewer"] = reviewer
        review["approved_at"] = datetime.now(timezone.utc).isoformat()
        draft["review"] = review
        write_yaml(draft_path, draft)
        promote_draft(
            draft_path,
            reviews_dir,
            atomic_dir,
            pinned_atomic_dir,
        )
        counts["promoted"] += 1
    return counts


def prepare(args: argparse.Namespace) -> None:
    atomic_dir = sync_atomic(args.checkout, args.revision)
    candidates = catalog_candidates(
        atomic_dir,
        DEFAULT_REVIEWS_DIR,
        DEFAULT_DRAFTS_DIR,
    )
    write_jsonl(DEFAULT_CATALOG_PATH, candidates)
    accepted, rejected, failed = draft_candidates(
        candidates,
        DEFAULT_DRAFTS_DIR,
        count=args.count,
        profile=args.profile,
        region=args.region,
        model_id=args.distiller_model_id,
        reviewer_model_id=args.reviewer_model_id,
        reviewer_fallback_model_id=args.reviewer_fallback_model_id or None,
        reviewer_min_confidence=args.reviewer_min_confidence,
        revision=args.revision,
        structured_retries=args.structured_retries,
    )
    queue_counts = write_review_queue(
        DEFAULT_DRAFTS_DIR,
        atomic_dir,
        REVIEW_QUEUE_PATH,
    )
    remaining = catalog_candidates(
        atomic_dir,
        DEFAULT_REVIEWS_DIR,
        DEFAULT_DRAFTS_DIR,
    )
    write_jsonl(DEFAULT_CATALOG_PATH, remaining)
    print(
        f"Prepared {accepted} accepted and {rejected} rejected drafts; "
        f"{failed} failed and remain eligible for retry; "
        f"{len(remaining)} candidates remain. Review {REVIEW_QUEUE_PATH} "
        f"({queue_counts['accepted']} accepted, "
        f"{queue_counts['rejected']} rejected total)."
    )


def promote(args: argparse.Namespace) -> None:
    if not args.confirm_reviewed:
        raise ValueError(
            "--confirm-reviewed is required to attest that the queue was reviewed"
        )
    atomic_dir = args.checkout / "atomics"
    if not atomic_dir.exists():
        raise FileNotFoundError("run `workflow.py prepare` before promotion")
    counts = batch_promote(
        drafts_dir=DEFAULT_DRAFTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        atomic_dir=atomic_dir,
        pinned_atomic_dir=DEFAULT_PINNED_ATOMIC_DIR,
        reviewer=args.reviewer,
        min_confidence=args.reviewer_min_confidence,
    )
    print(
        f"Promoted {counts['promoted']} drafts; "
        f"{counts['already_promoted']} already promoted; "
        f"{counts['ineligible']} ineligible."
    )


def benchmark(args: argparse.Namespace) -> None:
    build_outputs(
        DEFAULT_PINNED_ATOMIC_DIR,
        DEFAULT_REVIEWS_DIR,
        GENERATED_DIR,
    )
    settings = {
        role: ModelSettings.from_environment(role)
        for role in ("generator", "target", "oracle")
    }
    fallback_settings = {
        role: ModelSettings.fallback_from_environment(role)
        for role in ("generator", "target", "oracle")
    }
    if (
        args.max_attempts > 0
        and not args.allow_correlated_oracle
        and settings["target"].model_identity()
        == settings["oracle"].model_identity()
    ):
        raise ValueError(
            "target and oracle must use different provider/model identities"
        )

    if not args.resume and ATTEMPTS_PATH.exists():
        ATTEMPTS_PATH.unlink()
    if not args.resume and BENCHMARK_ERRORS_PATH.exists():
        BENCHMARK_ERRORS_PATH.unlink()
    curated = load_jsonl(GENERATED_DIR / "classifier_examples.jsonl")
    _, tasks, eval_episodes = split_tasks_by_episode(
        load_tasks(GENERATED_DIR / "generator_tasks.jsonl"),
        curated,
        args.eval_fraction,
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    attempt_count = 0
    success_count = 0
    if args.max_attempts > 0:
        for task in tasks:
            records = run_adversarial_task(
                task=task,
                classifier_view=args.classifier_view,
                max_attempts=args.max_attempts,
                oracle_min_confidence=args.oracle_min_confidence,
                log_path=ATTEMPTS_PATH,
                settings=settings,
                fallback_settings=fallback_settings,
                structured_retries=args.structured_retries,
                error_path=BENCHMARK_ERRORS_PATH,
            )
            attempt_count += len(records)
            success_count += sum(record["success"] for record in records)

    attempts = (
        load_jsonl(ATTEMPTS_PATH, missing_ok=True)
        if args.max_attempts > 0 or args.resume
        else []
    )
    examples, export_stats = render_training_examples(
        curated,
        attempts,
        oracle_min_confidence=args.oracle_min_confidence,
        allow_correlated_oracle=args.allow_correlated_oracle,
        eval_fraction=args.eval_fraction,
    )
    write_jsonl(TRAINING_PATH, examples)
    benchmark_examples = [
        example
        for example in examples
        if example["split"] == "eval"
    ]
    write_jsonl(BENCHMARK_PATH, benchmark_examples)
    evaluation_errors = load_jsonl(
        BENCHMARK_ERRORS_PATH,
        missing_ok=True,
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "atomic_revision": ATOMIC_REVISION,
        "review_count": len(list(DEFAULT_REVIEWS_DIR.glob("*.yaml"))),
        "task_count": len(tasks),
        "eval_episode_count": len(eval_episodes),
        "attempt_count": attempt_count,
        "successful_bypasses": success_count,
        "structured_evaluation_error_count": len(evaluation_errors),
        "training_example_count": len(examples),
        "benchmark_example_count": len(benchmark_examples),
        "classifier_view": args.classifier_view,
        "max_attempts": args.max_attempts,
        "models": {
            role: value.public_metadata()
            for role, value in settings.items()
        },
        "fallback_models": {
            role: (
                value.public_metadata()
                if value is not None
                else None
            )
            for role, value in fallback_settings.items()
        },
        "export_stats": export_stats,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(examples)} training examples and "
        f"{len(benchmark_examples)} held-out benchmark examples from "
        f"{attempt_count} attempts ({success_count} bypasses, "
        f"{len(evaluation_errors)} structured evaluation errors)."
    )


def augment(args: argparse.Namespace) -> None:
    if args.max_attempts < 2:
        raise ValueError("augmentation requires at least 2 attempts")
    if args.trajectories_per_task < 1:
        raise ValueError("trajectories_per_task must be at least 1")

    build_outputs(
        DEFAULT_PINNED_ATOMIC_DIR,
        DEFAULT_REVIEWS_DIR,
        GENERATED_DIR,
    )
    settings = {
        role: ModelSettings.from_environment(role)
        for role in ("generator", "target", "oracle")
    }
    fallback_settings = {
        role: ModelSettings.fallback_from_environment(role)
        for role in ("generator", "target", "oracle")
    }
    if (
        not args.allow_correlated_oracle
        and settings["target"].model_identity()
        == settings["oracle"].model_identity()
    ):
        raise ValueError(
            "target and oracle must use different provider/model identities"
        )

    curated = load_jsonl(GENERATED_DIR / "classifier_examples.jsonl")
    tasks, _, eval_episodes = split_tasks_by_episode(
        load_tasks(GENERATED_DIR / "generator_tasks.jsonl"),
        curated,
        args.eval_fraction,
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    if not args.resume and AUGMENT_ATTEMPTS_PATH.exists():
        AUGMENT_ATTEMPTS_PATH.unlink()
    if not args.resume and AUGMENT_ERRORS_PATH.exists():
        AUGMENT_ERRORS_PATH.unlink()

    new_attempt_count = 0
    new_success_count = 0
    for classifier_view in args.views:
        for task in tasks:
            for trajectory_index in range(1, args.trajectories_per_task + 1):
                records = run_adversarial_task(
                    task=task,
                    classifier_view=classifier_view,
                    max_attempts=args.max_attempts,
                    oracle_min_confidence=args.oracle_min_confidence,
                    log_path=AUGMENT_ATTEMPTS_PATH,
                    settings=settings,
                    fallback_settings=fallback_settings,
                    structured_retries=args.structured_retries,
                    error_path=AUGMENT_ERRORS_PATH,
                    stop_on_success=False,
                    trajectory_index=trajectory_index,
                )
                new_attempt_count += len(records)
                new_success_count += sum(
                    record["success"]
                    for record in records
                )

    attempts = load_jsonl(AUGMENT_ATTEMPTS_PATH, missing_ok=True)
    examples, export_stats = render_training_examples(
        curated,
        attempts,
        oracle_min_confidence=args.oracle_min_confidence,
        allow_correlated_oracle=args.allow_correlated_oracle,
        eval_fraction=args.eval_fraction,
    )
    write_jsonl(TRAINING_PATH, examples)
    augmented_training = [
        example
        for example in examples
        if example["split"] == "train"
    ]
    augmentation_examples = [
        example
        for example in augmented_training
        if example["source"]["kind"]
        == "oracle_preserved_adversarial_attempt"
    ]
    write_jsonl(AUGMENTED_TRAIN_PATH, augmented_training)
    write_jsonl(AUGMENT_EXAMPLES_PATH, augmentation_examples)
    evaluation_errors = load_jsonl(
        AUGMENT_ERRORS_PATH,
        missing_ok=True,
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "atomic_revision": ATOMIC_REVISION,
        "review_count": len(list(DEFAULT_REVIEWS_DIR.glob("*.yaml"))),
        "training_task_count": len(tasks),
        "excluded_eval_episode_count": len(eval_episodes),
        "views": args.views,
        "trajectories_per_task": args.trajectories_per_task,
        "max_attempts": args.max_attempts,
        "new_attempt_count": new_attempt_count,
        "total_attempt_count": len(attempts),
        "successful_bypass_count": new_success_count,
        "structured_evaluation_error_count": len(evaluation_errors),
        "combined_training_example_count": len(augmented_training),
        "generated_training_example_count": len(augmentation_examples),
        "models": {
            role: value.public_metadata()
            for role, value in settings.items()
        },
        "fallback_models": {
            role: (
                value.public_metadata()
                if value is not None
                else None
            )
            for role, value in fallback_settings.items()
        },
        "export_stats": export_stats,
    }
    AUGMENT_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(augmentation_examples)} generated examples and "
        f"{len(augmented_training)} combined training examples from "
        f"{len(attempts)} total attempts "
        f"({len(evaluation_errors)} structured evaluation errors)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command orchestration for classifier dataset generation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--count", type=int, default=25)
    prepare_parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    prepare_parser.add_argument("--revision", default=ATOMIC_REVISION)
    prepare_parser.add_argument("--profile", default=DEFAULT_AWS_PROFILE)
    prepare_parser.add_argument("--region", default=DEFAULT_AWS_REGION)
    prepare_parser.add_argument(
        "--distiller-model-id",
        default=DEFAULT_DISTILLER_MODEL,
    )
    prepare_parser.add_argument(
        "--reviewer-model-id",
        default=DEFAULT_REVIEWER_MODEL,
    )
    prepare_parser.add_argument(
        "--reviewer-fallback-model-id",
        default=DEFAULT_REVIEWER_FALLBACK_MODEL,
    )
    prepare_parser.add_argument(
        "--reviewer-min-confidence",
        type=float,
        default=0.8,
    )
    prepare_parser.add_argument(
        "--structured-retries",
        type=int,
        default=3,
    )
    prepare_parser.set_defaults(handler=prepare)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--reviewer", required=True)
    promote_parser.add_argument("--confirm-reviewed", action="store_true")
    promote_parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    promote_parser.add_argument(
        "--reviewer-min-confidence",
        type=float,
        default=0.8,
    )
    promote_parser.set_defaults(handler=promote)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--max-attempts", type=int, default=8)
    benchmark_parser.add_argument("--limit", type=int)
    benchmark_parser.add_argument(
        "--classifier-view",
        choices=("command_only", "last_user", "minimal_user", "all_user"),
        default="minimal_user",
    )
    benchmark_parser.add_argument(
        "--oracle-min-confidence",
        type=float,
        default=0.75,
    )
    benchmark_parser.add_argument("--eval-fraction", type=float, default=0.2)
    benchmark_parser.add_argument("--resume", action="store_true")
    benchmark_parser.add_argument(
        "--structured-retries",
        type=int,
        default=3,
    )
    benchmark_parser.add_argument(
        "--allow-correlated-oracle",
        action="store_true",
    )
    benchmark_parser.set_defaults(handler=benchmark)

    augment_parser = subparsers.add_parser("augment")
    augment_parser.add_argument("--trajectories-per-task", type=int, default=3)
    augment_parser.add_argument("--max-attempts", type=int, default=8)
    augment_parser.add_argument("--limit", type=int)
    augment_parser.add_argument(
        "--views",
        nargs="+",
        choices=("command_only", "last_user", "minimal_user", "all_user"),
        default=["minimal_user"],
    )
    augment_parser.add_argument(
        "--oracle-min-confidence",
        type=float,
        default=0.75,
    )
    augment_parser.add_argument("--eval-fraction", type=float, default=0.2)
    augment_parser.add_argument("--resume", action="store_true")
    augment_parser.add_argument(
        "--structured-retries",
        type=int,
        default=3,
    )
    augment_parser.add_argument(
        "--allow-correlated-oracle",
        action="store_true",
    )
    augment_parser.set_defaults(handler=augment)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
