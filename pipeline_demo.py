"""
LoopKeeper — end-to-end demo pipeline.

Runs four weekly meetings (real dates, chronological) through the full
system: extraction -> dedup -> state update -> workload analytics. No
network access or API key required — Module 1 (extraction) is stood in by
`extraction_module.extract_offline`, which produces schema-shaped output
from hand-authored dicts below (see extraction_module.py's docstring for
why that's a fair stand-in). Modules 2-5 run exactly as they would in
production.

Run:  python3 pipeline_demo.py
"""

from __future__ import annotations

from datetime import date

from dedup_engine import DedupEngine
from extraction_module import extract_offline
from models import StateStore
from state_engine import apply_overdue_check, process_meeting
from workload_analytics import compute_workload, format_report_text, to_json

# ---------------------------------------------------------------------------
# Demo dataset: 4 weekly meetings for a 3-person team.
# Each meeting's `items` list is what a real extraction call (Module 1)
# would have returned for that meeting's raw notes.
# ---------------------------------------------------------------------------

MEETINGS = [
    dict(
        meeting_date="2026-08-20", title="Weekly Sync",
        items=[
            dict(assignee="Aditi", action_description="Draft the Q4 investor pitch deck",
                 deadline="2026-08-28"),
            dict(assignee="Rohan", action_description="Set up staging environment for the API",
                 deadline="2026-08-25"),
            dict(assignee="Rohan", action_description="Update API documentation for v2 endpoints",
                 deadline="2026-08-24"),
            dict(assignee="Farah", action_description="Get budget sign-off from finance",
                 deadline="2026-08-22"),
        ],
    ),
    dict(
        meeting_date="2026-08-27", title="Weekly Sync",
        items=[
            dict(assignee="Aditi", action_description="Finish the investor pitch deck slides",
                 deadline="2026-09-04", pushback_signal=True,
                 notes="Ran out of time this week, pushing a week."),
            dict(assignee="Rohan", action_description="Staging environment for the API is live",
                 status_explicit="done"),
            dict(assignee="Rohan", action_description="Write integration tests for the API",
                 deadline="2026-09-10"),
            # Farah's budget sign-off task isn't mentioned at all this meeting —
            # its 2026-08-22 deadline has already passed, so the automatic
            # overdue sweep (not an explicit statement) should flip it.
        ],
    ),
    dict(
        meeting_date="2026-09-03", title="Weekly Sync",
        items=[
            dict(assignee="Aditi", action_description="Finish the investor pitch deck slides",
                 deadline="2026-09-11", pushback_signal=True,
                 notes="Still waiting on design assets."),
            dict(assignee="Farah", action_description="Budget sign-off from finance is confirmed",
                 status_explicit="done"),
            dict(assignee="Farah", action_description="Prepare vendor contract redlines",
                 deadline="2026-09-08"),
            dict(assignee="Rohan", action_description="Continue integration tests for the API",
                 is_followup_reference=True,
                 notes="In progress, restated with no change."),
        ],
    ),
    dict(
        meeting_date="2026-09-10", title="Weekly Sync",
        items=[
            dict(assignee="Aditi", action_description="Complete the investor pitch deck for the board",
                 deadline="2026-09-18", pushback_signal=True,
                 notes="Third delay — now blocked on exec review availability."),
            dict(assignee="Rohan", action_description="Integration tests for the API are complete",
                 status_explicit="done"),
            dict(assignee="Rohan", action_description="Deploy API to production",
                 deadline="2026-09-16"),
            # Farah's vendor contract redlines task (due 2026-09-08) isn't
            # mentioned — already overdue by this meeting's date; sweep confirms it.
        ],
    ),
]

TODAY = date(2026, 9, 17)  # "as of today" snapshot for the final workload report


def run() -> None:
    store = StateStore()
    engine = DedupEngine(store)  # default: offline HashingTfidfBackend, no network needed

    print("=" * 84)
    print("LOOPKEEPER — PROCESSING MEETINGS IN CHRONOLOGICAL ORDER")
    print("=" * 84)

    last_meeting = None
    for m in MEETINGS:
        extraction = extract_offline(m["meeting_date"], m["title"], m["items"])
        meeting = store.add_meeting(
            meeting_date=date.fromisoformat(extraction["meeting_date"]),
            raw_notes=f"[offline demo — {len(m['items'])} items authored directly, see pipeline_demo.py]",
            title=extraction["meeting_title"],
            attendees=extraction["attendees"],
        )
        summary = process_meeting(store, engine, meeting, extraction["action_items"])
        last_meeting = meeting

        print(f"\n--- {meeting.meeting_date}  \"{meeting.title}\"  "
              f"(created={summary.created}, updated={summary.updated}, "
              f"pushbacks={summary.pushbacks_detected}, status_changes={summary.status_changes}, "
              f"auto_overdue={summary.auto_overdue}) ---")
        for line in summary.log:
            print("   " + line)

    # Standalone "as of today" sweep — catches anything that crossed its
    # deadline between the last meeting and now, without needing a new meeting.
    flipped_today = apply_overdue_check(store, as_of=TODAY, meeting_id=last_meeting.id)
    print(f"\n--- Standalone overdue sweep as of {TODAY} ---")
    print(f"   Flipped {flipped_today} item(s) pending -> overdue since the last meeting.")

    # -----------------------------------------------------------------
    # Audit trail example: full history of Aditi's pitch-deck task, to
    # show the pushback pattern the system caught across 4 meetings.
    # -----------------------------------------------------------------
    pitch_item = next(ai for ai in store.action_items.values() if "pitch deck" in ai.description.lower())
    print(f"\n--- Full audit trail for action_item #{pitch_item.id}: \"{pitch_item.description}\" ---")
    print(f"   pushback_count={pitch_item.pushback_count}  is_chronic_delay={pitch_item.is_chronic_delay}  "
          f"status={pitch_item.status.value}  current_deadline={pitch_item.current_deadline}")
    for event in store.history_for(pitch_item.id):
        meeting = store.meetings[event.meeting_id]
        print(f"   [{meeting.meeting_date}] {event.event_type.value}: "
              f"{event.old_value} -> {event.new_value}"
              + (f"  ({event.note})" if event.note else ""))

    # -----------------------------------------------------------------
    # Workload analytics
    # -----------------------------------------------------------------
    report = compute_workload(store, as_of=TODAY)
    print("\n" + format_report_text(report))

    print("\n--- JSON form (what a dashboard/API would consume) ---")
    import json
    print(json.dumps(to_json(report), indent=2))


if __name__ == "__main__":
    run()
