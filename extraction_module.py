"""
LoopKeeper — Module 1: Extraction wiring.

This module shows how raw meeting notes get turned into the structured JSON
defined by extraction_schema.json, in two forms:

  1. `extract_with_claude(...)`  — the real implementation: calls Claude with
     the system prompt from extraction_prompt.md and forces schema-
     conformant JSON output via tool-use (a "structured output" tool whose
     input_schema IS extraction_schema.json, so the model has no choice but
     to emit a matching object). Requires ANTHROPIC_API_KEY; not executed by
     the offline demo pipeline.

  2. `extract_offline(...)`      — a deterministic stand-in used by
     pipeline_demo.py so the rest of the system (dedup, state, analytics)
     can be demonstrated end-to-end without network access or an API key.
     It's intentionally "dumb" (regex/keyword based) — it exists only to
     produce schema-shaped output from the demo transcripts, not as a
     serious extraction strategy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).parent / "extraction_schema.json"
PROMPT_PATH = Path(__file__).parent / "extraction_prompt.md"


def extract_with_claude(meeting_date: str, raw_notes: str, meeting_title: str | None = None,
                         model: str = "claude-sonnet-4-5") -> dict[str, Any]:
    """
    Real extraction path. Uses Anthropic's tool-use feature to force the
    response to validate against extraction_schema.json — the model is
    given exactly one tool ("record_action_items") whose input_schema is
    the JSON schema, and `tool_choice` forces it to call that tool, so the
    "output" is the tool call's structured input rather than free text that
    needs separate JSON parsing/repair.

        export ANTHROPIC_API_KEY=...
        result = extract_with_claude("2026-09-17", raw_notes)

    Left uncalled by the demo pipeline (no API key assumed in this
    environment) but this is the production wiring.
    """
    import anthropic  # pip install anthropic

    schema = json.loads(SCHEMA_PATH.read_text())
    system_prompt = _load_system_prompt()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        tools=[{
            "name": "record_action_items",
            "description": "Record the structured action items extracted from this meeting.",
            "input_schema": schema,
        }],
        tool_choice={"type": "tool", "name": "record_action_items"},
        messages=[{
            "role": "user",
            "content": (
                f"meeting_date: {meeting_date}\n"
                f"meeting_title: {meeting_title or 'null'}\n\n"
                f'raw_notes:\n"""\n{raw_notes}\n"""\n\n'
                "Return JSON matching the LoopKeeperMeetingExtraction schema."
            ),
        }],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_action_items":
            return block.input  # already validated to match the schema by the API

    raise RuntimeError("Model did not return a record_action_items tool call.")


def _load_system_prompt() -> str:
    text = PROMPT_PATH.read_text()
    start = text.index("## SYSTEM PROMPT")
    fenced = text[start:].split("```", 2)
    return fenced[1].strip() if len(fenced) > 1 else text


# ---------------------------------------------------------------------------
# Offline stand-in for the demo pipeline (no LLM call, no network)
# ---------------------------------------------------------------------------

def extract_offline(meeting_date: str, meeting_title: str, items: list[dict]) -> dict[str, Any]:
    """
    Builds a schema-shaped extraction result directly from hand-authored
    item dicts (see pipeline_demo.py). This simulates "what the LLM would
    have returned" for a fixed, reviewable demo dataset — it applies the
    same defaulting rules an LLM response would already satisfy (fills in
    optional fields), so downstream modules never need to know the
    difference between this and a real extraction.
    """
    action_items = []
    for raw in items:
        action_items.append({
            "raw_text": raw.get("raw_text", raw["action_description"]),
            "action_description": raw["action_description"],
            "assignee": raw["assignee"],
            "deadline": raw.get("deadline"),
            "deadline_phrase_raw": raw.get("deadline_phrase_raw"),
            "status_explicit": raw.get("status_explicit", "none"),
            "is_followup_reference": raw.get("is_followup_reference", False),
            "pushback_signal": raw.get("pushback_signal", False),
            "notes": raw.get("notes"),
            "confidence": raw.get("confidence", 0.95),
        })

    return {
        "meeting_date": meeting_date,
        "meeting_title": meeting_title,
        "attendees": raw_attendees(items),
        "action_items": action_items,
    }


def raw_attendees(items: list[dict]) -> list[str]:
    seen: list[str] = []
    for it in items:
        name = it.get("assignee")
        if name and name not in seen and name != "Unassigned":
            seen.append(name)
    return seen
