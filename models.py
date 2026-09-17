"""
LoopKeeper — Shared domain models and in-memory state store.

Every other module (dedup_engine, state_engine, workload_analytics) operates
against the `StateStore` defined here rather than talking to a database
directly. That keeps the core logic testable and DB-agnostic: swapping this
for a real Postgres-backed store means implementing the same methods with
INSERT/UPDATE/SELECT against schema.sql instead of dict operations — the call
sites in the other modules don't change. Each method below notes the SQL
statement it stands in for.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums (mirror the Postgres ENUM types in schema.sql)
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PENDING = "pending"
    DONE = "done"
    OVERDUE = "overdue"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    CREATED = "created"
    STATUS_CHANGE = "status_change"
    DEADLINE_PUSHBACK = "deadline_pushback"
    REASSIGNED = "reassigned"
    REPHRASED = "rephrased"
    MERGED = "merged"


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@dataclass
class Assignee:
    id: int
    canonical_name: str
    email: Optional[str] = None
    aliases: list[str] = field(default_factory=list)


@dataclass
class Meeting:
    id: int
    meeting_date: date
    title: Optional[str] = None
    raw_notes: str = ""
    attendees: list[str] = field(default_factory=list)


@dataclass
class ActionItem:
    id: int
    assignee_id: int
    description: str
    action_core: str
    embedding: Optional[np.ndarray]
    original_deadline: Optional[date]
    current_deadline: Optional[date]
    pushback_count: int
    is_chronic_delay: bool
    status: Status
    first_meeting_id: int
    last_meeting_id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class TaskHistoryEvent:
    id: int
    action_item_id: int
    meeting_id: int
    event_type: EventType
    old_value: dict
    new_value: dict
    note: Optional[str]
    recorded_at: datetime


# ---------------------------------------------------------------------------
# In-memory state store
# ---------------------------------------------------------------------------

class StateStore:
    """
    DB-agnostic persistent state. Backed by dicts here; a production
    implementation swaps each method's body for SQL against `schema.sql`
    while keeping the same signatures.
    """

    def __init__(self, chronic_pushback_threshold: int = 3):
        self._assignee_ids = itertools.count(1)
        self._meeting_ids = itertools.count(1)
        self._item_ids = itertools.count(1)
        self._history_ids = itertools.count(1)

        self.assignees: dict[int, Assignee] = {}
        self.meetings: dict[int, Meeting] = {}
        self.action_items: dict[int, ActionItem] = {}
        self.task_history: list[TaskHistoryEvent] = []

        self._name_index: dict[str, int] = {}  # lowercased name/alias -> assignee_id
        self.chronic_pushback_threshold = chronic_pushback_threshold

    # -- assignees -----------------------------------------------------
    # SQL equiv: SELECT ... WHERE canonical_name ILIKE %s OR %s = ANY(aliases);
    #            INSERT INTO assignees (...) ON CONFLICT DO NOTHING;
    def get_or_create_assignee(self, name_as_stated: str) -> Assignee:
        key = name_as_stated.strip().lower()
        if key in self._name_index:
            return self.assignees[self._name_index[key]]

        assignee = Assignee(id=next(self._assignee_ids), canonical_name=name_as_stated.strip())
        self.assignees[assignee.id] = assignee
        self._name_index[key] = assignee.id
        return assignee

    def add_alias(self, assignee_id: int, alias: str) -> None:
        # SQL equiv: UPDATE assignees SET aliases = array_append(aliases, %s) WHERE id = %s;
        key = alias.strip().lower()
        if key not in self._name_index:
            self.assignees[assignee_id].aliases.append(alias.strip())
            self._name_index[key] = assignee_id

    # -- meetings --------------------------------------------------------
    # SQL equiv: INSERT INTO meetings (meeting_date, title, raw_notes, attendees) VALUES (...);
    def add_meeting(self, meeting_date: date, raw_notes: str, title: Optional[str] = None,
                     attendees: Optional[list[str]] = None) -> Meeting:
        meeting = Meeting(
            id=next(self._meeting_ids),
            meeting_date=meeting_date,
            title=title,
            raw_notes=raw_notes,
            attendees=attendees or [],
        )
        self.meetings[meeting.id] = meeting
        return meeting

    # -- action items ------------------------------------------------------
    # SQL equiv: SELECT * FROM action_items WHERE assignee_id = %s AND is_active;
    def active_items_for_assignee(self, assignee_id: int) -> list[ActionItem]:
        return [ai for ai in self.action_items.values()
                if ai.assignee_id == assignee_id and ai.is_active]

    # SQL equiv: SELECT * FROM action_items WHERE is_active;  (used for cross-assignee
    #            reassignment checks, where the dedup engine widens the search)
    def all_active_items(self) -> list[ActionItem]:
        return [ai for ai in self.action_items.values() if ai.is_active]

    # SQL equiv: INSERT INTO action_items (...) VALUES (...); + INSERT INTO task_history (...)
    def create_action_item(self, *, assignee_id: int, description: str, action_core: str,
                            embedding: Optional[np.ndarray], deadline: Optional[date],
                            meeting_id: int, status: Status = Status.PENDING) -> ActionItem:
        now = datetime.now(timezone.utc)
        item = ActionItem(
            id=next(self._item_ids),
            assignee_id=assignee_id,
            description=description,
            action_core=action_core,
            embedding=embedding,
            original_deadline=deadline,
            current_deadline=deadline,
            pushback_count=0,
            is_chronic_delay=False,
            status=status,
            first_meeting_id=meeting_id,
            last_meeting_id=meeting_id,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        self.action_items[item.id] = item
        self.log_history(
            action_item_id=item.id, meeting_id=meeting_id, event_type=EventType.CREATED,
            old_value={}, new_value={"description": description, "deadline": _iso(deadline),
                                      "assignee_id": assignee_id, "status": status.value},
            note="First observed.",
        )
        return item

    # SQL equiv: UPDATE action_items SET ... WHERE id = %s; + INSERT INTO task_history (...)
    def log_history(self, *, action_item_id: int, meeting_id: int, event_type: EventType,
                     old_value: dict, new_value: dict, note: Optional[str] = None) -> TaskHistoryEvent:
        event = TaskHistoryEvent(
            id=next(self._history_ids),
            action_item_id=action_item_id,
            meeting_id=meeting_id,
            event_type=event_type,
            old_value=old_value,
            new_value=new_value,
            note=note,
            recorded_at=datetime.now(timezone.utc),
        )
        self.task_history.append(event)
        return event

    def history_for(self, action_item_id: int) -> list[TaskHistoryEvent]:
        return [e for e in self.task_history if e.action_item_id == action_item_id]


def _iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None
