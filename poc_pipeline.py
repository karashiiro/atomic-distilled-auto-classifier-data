import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent
TOKEN_PATTERN = re.compile(r"#\{([^}]+)\}")
VIEW_NAMES = ("command_only", "last_user", "minimal_user", "all_user")
DECISION_LABELS = {"allow", "deny", "request_context"}
HARM_LABELS = {"benign", "harmful", "context_dependent"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return document


def load_atomic_tests(atomic_dir: Path) -> dict[str, dict[str, Any]]:
    tests: dict[str, dict[str, Any]] = {}
    for path in sorted(atomic_dir.rglob("*.yaml")):
        document = load_yaml(path)
        technique_id = document.get("attack_technique")
        if technique_id is None or not isinstance(
            document.get("atomic_tests"),
            list,
        ):
            continue
        for test in document.get("atomic_tests", []):
            guid = test["auto_generated_guid"]
            if guid in tests:
                raise ValueError(f"duplicate Atomic test GUID: {guid}")
            tests[guid] = {
                **test,
                "attack_technique": technique_id,
                "source_path": path,
            }
    return tests


def resolve_command(test: dict[str, Any], review: dict[str, Any]) -> str:
    arguments = test.get("input_arguments", {})
    values = {
        name: definition.get("default")
        for name, definition in arguments.items()
    }
    values.update(review.get("argument_values", {}))

    command = test["executor"]["command"].strip()

    def replace_token(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values or values[name] is None:
            raise ValueError(f"no value supplied for Atomic argument {name!r}")
        return str(values[name])

    command = TOKEN_PATTERN.sub(replace_token, command)
    for old, new in review.get("literal_replacements", {}).items():
        command = command.replace(old, new)
    if TOKEN_PATTERN.search(command):
        raise ValueError("resolved command still contains Atomic argument tokens")
    return command


def validate_context(context: dict[str, Any]) -> None:
    if context["harm_label"] not in HARM_LABELS:
        raise ValueError(f"invalid harm label in context {context['id']}")

    messages = context["user_messages"]
    if not messages or not all(isinstance(message, str) for message in messages):
        raise ValueError(f"context {context['id']} needs user messages")

    indices = context["minimal_user_message_indices"]
    if not indices or any(index < 0 or index >= len(messages) for index in indices):
        raise ValueError(f"context {context['id']} has invalid minimal-message indices")

    labels = context["view_labels"]
    if set(labels) != set(VIEW_NAMES):
        raise ValueError(f"context {context['id']} must label every classifier view")
    if not set(labels.values()) <= DECISION_LABELS:
        raise ValueError(f"context {context['id']} has an invalid decision label")


def build_episode(test: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    source = review["source"]
    if source["atomic_guid"] != test["auto_generated_guid"]:
        raise ValueError("review and Atomic test GUIDs do not match")
    if source["technique_id"] != test["attack_technique"]:
        raise ValueError("review and Atomic technique IDs do not match")

    for context in review["contexts"]:
        validate_context(context)

    executor = test["executor"]
    return {
        "schema_version": 1,
        "episode_id": review["id"],
        "source": {
            **source,
            "atomic_name": test["name"],
            "atomic_description": test["description"].strip(),
        },
        "execution": {
            "platforms": test["supported_platforms"],
            "executor": executor["name"],
            "elevation_required": executor.get("elevation_required", False),
            "command": resolve_command(test, review),
            "prerequisites": test.get("dependencies", []),
        },
        "behavior": review["behavior"],
        "contexts": review["contexts"],
    }


def build_episodes(atomic_dir: Path, reviews_dir: Path) -> list[dict[str, Any]]:
    atomic_tests = load_atomic_tests(atomic_dir)
    episodes = []
    for review_path in sorted(reviews_dir.glob("*.yaml")):
        review = load_yaml(review_path)
        guid = review["source"]["atomic_guid"]
        if guid not in atomic_tests:
            raise ValueError(f"{review_path} references unknown Atomic GUID {guid}")
        episodes.append(build_episode(atomic_tests[guid], review))
    return episodes


def render_messages(context: dict[str, Any], view: str) -> list[dict[str, str]]:
    messages = context["user_messages"]
    if view == "command_only":
        selected: list[str] = []
    elif view == "last_user":
        selected = messages[-1:]
    elif view == "minimal_user":
        selected = [
            messages[index]
            for index in context["minimal_user_message_indices"]
        ]
    elif view == "all_user":
        selected = messages
    else:
        raise ValueError(f"unknown classifier view: {view}")
    return [{"role": "user", "content": message} for message in selected]


def render_classifier_examples(
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples = []
    for episode in episodes:
        for context in episode["contexts"]:
            for view in VIEW_NAMES:
                examples.append(
                    {
                        "schema_version": 1,
                        "episode_id": episode["episode_id"],
                        "context_id": context["id"],
                        "view": view,
                        "messages": render_messages(context, view),
                        "command": episode["execution"]["command"],
                        "scenario_harm_label": context["harm_label"],
                        "decision_label": context["view_labels"][view],
                    }
                )
    validate_visible_input_labels(examples)
    return examples


def validate_visible_input_labels(examples: list[dict[str, Any]]) -> None:
    labels_by_input: dict[str, set[str]] = {}
    for example in examples:
        visible_input = json.dumps(
            {
                "messages": example["messages"],
                "command": example["command"],
            },
            sort_keys=True,
        )
        labels_by_input.setdefault(visible_input, set()).add(
            example["decision_label"]
        )

    conflicts = [
        labels
        for labels in labels_by_input.values()
        if len(labels) > 1
    ]
    if conflicts:
        raise ValueError(
            "identical classifier inputs have contradictory decision labels"
        )


def render_generator_tasks(
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks = []
    for episode in episodes:
        for context in episode["contexts"]:
            if context["harm_label"] != "harmful":
                continue
            tasks.append(
                {
                    "schema_version": 1,
                    "episode_id": episode["episode_id"],
                    "context_id": context["id"],
                    "oracle_context": render_messages(context, "all_user"),
                    "classifier_views": {
                        view: {
                            "messages": render_messages(context, view),
                            "expected_decision": context["view_labels"][view],
                        }
                        for view in VIEW_NAMES
                    },
                    "seed_command": episode["execution"]["command"],
                    "execution": {
                        "platforms": episode["execution"]["platforms"],
                        "executor": episode["execution"]["executor"],
                    },
                    "behavior": episode["behavior"],
                }
            )
    return tasks


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_outputs(
    atomic_dir: Path,
    reviews_dir: Path,
    output_dir: Path,
) -> tuple[int, int, int]:
    episodes = build_episodes(atomic_dir, reviews_dir)
    classifier_examples = render_classifier_examples(episodes)
    generator_tasks = render_generator_tasks(episodes)

    write_jsonl(output_dir / "episodes.jsonl", episodes)
    write_jsonl(output_dir / "classifier_examples.jsonl", classifier_examples)
    write_jsonl(output_dir / "generator_tasks.jsonl", generator_tasks)
    return len(episodes), len(classifier_examples), len(generator_tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build context-sensitive classifier data from Atomic tests."
    )
    parser.add_argument(
        "--atomic-dir",
        type=Path,
        default=ROOT / "poc_data" / "atomic",
    )
    parser.add_argument(
        "--reviews-dir",
        type=Path,
        default=ROOT / "poc_data" / "reviews",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "generated",
    )
    args = parser.parse_args()

    counts = build_outputs(args.atomic_dir, args.reviews_dir, args.output_dir)
    print(
        f"Wrote {counts[0]} episodes, {counts[1]} classifier examples, "
        f"and {counts[2]} generator tasks."
    )


if __name__ == "__main__":
    main()
