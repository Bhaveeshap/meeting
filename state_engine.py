"""
LoopKeeper — Module 4: State Update Engine.

Consumes one meeting's worth of extracted action items (the output of
Module 1 / extraction_schema.json) and:
  1. Runs each item through the dedup engine (Module 2) to decide
     match-vs-new.
  2. Applies the resulting state transition to the persistent store:
       - creates a new action_item, or
       - updates the matched item's description/deadline/status,
     logging every change to task_history.
  3. Detects deadline pushbacks (Module 3) and increments pushback_count /
     flags chronic delay.
  4. After the meeting is processed, sweeps all active items and flips
     `pending` -> `overdue` for anything past its current_deadline as of
     this meeting's date (or an arbitrary `as_of` date for a live "today"
     snapshot).

This is the module you call once per meeting, in chronological order.
Processing meetings out of order will corrupt pushback/history semantics,
since "pushback" and "overdue" are both defined relative to what the engine
already believed at the time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from dedup_engine import DedupEngine, normalize_action_core
from models import ActionItem, EventType, Meeting, Status, StateStore

CHRONIC_PUSHBACK_THRESHOLD = 3  # pushback_count at/above this => flagged chronic


@dataclass
class ProcessingSummary:
    meeting_id: int
    created: int = 0
    updated: int = 0
    pushbacks_detected: int = 0
    status_changes: int = 0
    auto_overdue: int = 0
    log: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.log.append(msg)


_STATUS_MAP = {
    "done": Status.DONE,
    "pending": Status.PENDING,
    "blocked": Status.BLOCKED,
    "cancelled": Status.CANCELLED,
    "none": None,
}


def process_meeting(store: StateStore, engine: DedupEngine, meeting: Meeting,
                     extracted_items: list[dict]) -> ProcessingSummary:
    """
    `extracted_items` is a list of dicts shaped like the `action_items`
    entries in extraction_schema.json (already validated/parsed from the
    LLM's JSON output).
    """
    summary = ProcessingSummary(meeting_id=meeting.id)

    for raw_item in extracted_items:
        _process_single_item(store, engine, meeting, raw_item, summary)

    overdue_count = apply_overdue_check(store, as_of=meeting.meeting_date, meeting_id=meeting.id)
    summary.auto_overdue += overdue_count
    if overdue_count:
        summary.note(f"Auto-flagged {overdue_count} item(s) overdue as of {meeting.meeting_date}.")

    return summary


def _process_single_item(store: StateStore, engine: DedupEngine, meeting: Meeting,
                          raw_item: dict, summary: ProcessingSummary) -> None:
    assignee_name = raw_item["assignee"] or "Unassigned"
    description = raw_item["action_description"]
    deadline = _parse_date(raw_item.get("deadline"))
    status_explicit = _STATUS_MAP.get(raw_item.get("status_explicit", "none"))
    pushback_signal = bool(raw_item.get("pushback_signal", False))

    embedding = engine.embed(description)
    match = engine.find_match(assignee_name=assignee_name, description=description, embedding=embedding)

    if match.is_match and match.matched_item is not None:
        _apply_update(store, meeting, match.matched_item, description=description,
                       embedding=embedding, deadline=deadline, status_explicit=status_explicit,
                       pushback_signal=pushback_signal, summary=summary)
    else:
        assignee = store.get_or_create_assignee(assignee_name)
        item = store.create_action_item(
            assignee_id=assignee.id,
            description=description,
            action_core=normalize_action_core(description),
            embedding=embedding,
            deadline=deadline,
            meeting_id=meeting.id,
            status=status_explicit or Status.PENDING,
        )
        summary.created += 1
        summary.note(f"[NEW] #{item.id} '{description}' -> {assignee.canonical_name} "
                      f"(due {deadline or 'n/a'})")


def _apply_update(store: StateStore, meeting: Meeting, item: ActionItem, *, description: str,
                   embedding, deadline: Optional[date], status_explicit: Optional[Status],
                   pushback_signal: bool, summary: ProcessingSummary) -> None:
    changed = False

    # --- deadline / pushback handling -------------------------------------
    if deadline is not None and item.current_deadline is not None and deadline > item.current_deadline:
        old_deadline = item.current_deadline
        item.current_deadline = deadline
        item.pushback_count += 1
        if item.pushback_count >= CHRONIC_PUSHBACK_THRESHOLD:
            item.is_chronic_delay = True
        store.log_history(
            action_item_id=item.id, meeting_id=meeting.id, event_type=EventType.DEADLINE_PUSHBACK,
            old_value={"deadline": _iso(old_deadline)}, new_value={"deadline": _iso(deadline)},
            note="Explicit pushback signal in notes." if pushback_signal else "Deadline moved later.",
        )
        summary.pushbacks_detected += 1
        summary.note(f"[PUSHBACK] #{item.id} '{item.description}': {old_deadline} -> {deadline} "
                      f"(pushback_count={item.pushback_count}"
                      f"{', CHRONIC' if item.is_chronic_delay else ''})")
        changed = True
    elif deadline is not None and item.current_deadline is None:
        item.current_deadline = deadline  # first time a deadline is stated for this item
        changed = True
    elif deadline is not None and deadline < item.current_deadline:
        # Deadline moved EARLIER — not a pushback, but still track it (e.g. urgency escalation).
        store.log_history(
            action_item_id=item.id, meeting_id=meeting.id, event_type=EventType.STATUS_CHANGE,
            old_value={"deadline": _iso(item.current_deadline)}, new_value={"deadline": _iso(deadline)},
            note="Deadline moved earlier (escalated).",
        )
        item.current_deadline = deadline
        changed = True

    # --- status handling ----------------------------------------------------
    if status_explicit is not None and status_explicit != item.status:
        old_status = item.status
        item.status = status_explicit
        if status_explicit in (Status.DONE, Status.CANCELLED):
            item.is_active = False
        store.log_history(
            action_item_id=item.id, meeting_id=meeting.id, event_type=EventType.STATUS_CHANGE,
            old_value={"status": old_status.value}, new_value={"status": status_explicit.value},
        )
        summary.status_changes += 1
        summary.note(f"[STATUS] #{item.id} '{item.description}': {old_status.value} -> {status_explicit.value}")
        changed = True

    # --- description / rephrasing -------------------------------------------
    if description.strip().lower() != item.description.strip().lower():
        store.log_history(
            action_item_id=item.id, meeting_id=meeting.id, event_type=EventType.REPHRASED,
            old_value={"description": item.description}, new_value={"description": description},
            note="Updated to most recent phrasing; embedding refreshed.",
        )
        item.description = description
        item.action_core = normalize_action_core(description)
        item.embedding = embedding
        changed = True

    # The item was mentioned in this meeting regardless of whether any field
    # changed (e.g. a plain "still working on it" restatement) — always
    # advance last_meeting_id so "last seen" stays accurate for staleness
    # checks, but only count it toward `updated` when something substantive
    # actually changed.
    item.last_meeting_id = meeting.id
    if changed:
        item.updated_at = datetime.now(timezone.utc)
        summary.updated += 1
    else:
        summary.note(f"[SEEN] #{item.id} '{item.description}' restated, no field changes.")


def apply_overdue_check(store: StateStore, as_of: date, meeting_id: Optional[int] = None) -> int:
    """
    Sweep all active items and flip `pending` -> `overdue` where
    current_deadline < as_of. `meeting_id` ties the history entry to the
    triggering meeting; when running a standalone "as of today" snapshot
    (not tied to ingesting a specific meeting), pass the id of the most
    recent meeting instead so history stays attributable.
    """
    flipped = 0
    for item in store.all_active_items():
        if item.status == Status.PENDING and item.current_deadline and item.current_deadline < as_of:
            old_status = item.status
            item.status = Status.OVERDUE
            item.updated_at = datetime.now(timezone.utc)
            store.log_history(
                action_item_id=item.id,
                meeting_id=meeting_id or item.last_meeting_id,
                event_type=EventType.STATUS_CHANGE,
                old_value={"status": old_status.value},
                new_value={"status": Status.OVERDUE.value},
                note=f"Auto-detected: deadline {item.current_deadline} passed as of {as_of}.",
            )
            flipped += 1
    return flipped


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


def _iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None
