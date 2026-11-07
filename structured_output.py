import json
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel
from strands import Agent

T = TypeVar("T", bound=BaseModel)


class StructuredResponseError(RuntimeError):
    def __init__(self, role: str, diagnostics: list[dict[str, Any]]):
        super().__init__(
            f"{role} did not return valid structured output after "
            f"{len(diagnostics)} attempts"
        )
        self.role = role
        self.diagnostics = diagnostics


def parse_json_text(text: str, output_model: type[T]) -> T | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return output_model.model_validate(value)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def invoke_structured_resilient(
    agent_factory: Callable[[], Agent],
    prompt: str,
    output_model: type[T],
    *,
    role: str,
    max_attempts: int,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    diagnostics: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            attempt_prompt = prompt
            if attempt > 1:
                attempt_prompt += (
                    "\n\nA prior response failed schema validation. "
                    "Return the answer through the required structured-output "
                    "schema with every required field populated."
                )
            result = agent_factory()(
                attempt_prompt,
                structured_output_model=output_model,
            )
            if isinstance(result.structured_output, output_model):
                return result.structured_output

            text = str(result).strip()
            parsed = parse_json_text(text, output_model)
            if parsed is not None:
                return parsed
            diagnostics.append(
                {
                    "attempt": attempt,
                    "kind": "missing_structured_output",
                    "stop_reason": str(result.stop_reason),
                    "text": text[:8000],
                    "message": result.message,
                }
            )
            if str(result.stop_reason) == "content_filtered":
                break
        except Exception as error:
            diagnostics.append(
                {
                    "attempt": attempt,
                    "kind": "exception",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )

        if attempt < max_attempts:
            print(
                f"{role}: structured output attempt {attempt} failed; retrying"
            )
            time.sleep(min(2 ** (attempt - 1), 4))

    raise StructuredResponseError(role, diagnostics)
