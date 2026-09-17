"""
LoopKeeper — Module 5: Workload & Overload Analytics Engine.

Aggregates open/pending/overdue tasks per owner and flags people who look
overloaded, based on open task volume, overdue count, near-term deadlines,
and chronic-delay tasks. Produces both a machine-readable report (for the
dashboard / API) and a human-readable text table (for console/Slack output).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from models import Status, StateStore

# Weights used in the overload score. Overdue work counts more than merely
# open work; upcoming deadlines within a week add pressure; a history of
# chronic delay on top of current load compounds the risk.
WEIGHT_OPEN = 1.0
WEIGHT_OVERDUE = 2.5
WEIGHT_DUE_SOON = 1.5
WEIGHT_CHRONIC = 1.0

DUE_SOON_WINDOW_DAYS = 7


@dataclass
class WorkloadStats:
    assignee_id: int
    assignee_name: str
    open_count: int = 0
    overdue_count: int = 0
    done_count: int = 0
    blocked_count: int = 0
    chronic_delay_count: int = 0
    due_within_window_count: int = 0
    total_pushbacks: int = 0
    overload_score: float = 0.0
    overload_flag: bool = False


@dataclass
class WorkloadReport:
    as_of: date
    stats: list[WorkloadStats] = field(default_factory=list)
    team_mean_score: float = 0.0
    team_stdev_score: float = 0.0


def compute_workload(store: StateStore, as_of: date,
                      due_soon_window_days: int = DUE_SOON_WINDOW_DAYS) -> WorkloadReport:
    per_assignee: dict[int, WorkloadStats] = {
        a.id: WorkloadStats(assignee_id=a.id, assignee_name=a.canonical_name)
        for a in store.assignees.values()
    }

    horizon = as_of + timedelta(days=due_soon_window_days)

    for item in store.action_items.values():
        stats = per_assignee.get(item.assignee_id)
        if stats is None:
            continue  # defensive; shouldn't happen

        if item.status == Status.DONE:
            stats.done_count += 1
            continue
        if item.status == Status.CANCELLED:
            continue  # cancelled work doesn't count toward load either way

        if item.status == Status.PENDING:
            stats.open_count += 1
        elif item.status == Status.OVERDUE:
            stats.overdue_count += 1
            stats.open_count += 1  # overdue is still open work, just past-due
        elif item.status == Status.BLOCKED:
            stats.blocked_count += 1
            stats.open_count += 1

        if item.is_chronic_delay:
            stats.chronic_delay_count += 1
        stats.total_pushbacks += item.pushback_count

        if item.current_deadline and item.status in (Status.PENDING, Status.OVERDUE) \
                and as_of <= item.current_deadline <= horizon:
            stats.due_within_window_count += 1

    for stats in per_assignee.values():
        stats.overload_score = (
            WEIGHT_OPEN * stats.open_count
            + WEIGHT_OVERDUE * stats.overdue_count
            + WEIGHT_DUE_SOON * stats.due_within_window_count
            + WEIGHT_CHRONIC * stats.chronic_delay_count
        )

    scores = [s.overload_score for s in per_assignee.values()]
    mean_score = statistics.fmean(scores) if scores else 0.0
    stdev_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0

    # Flag overloaded: meaningfully above the team's own baseline (mean + stdev),
    # OR an absolute red-flag condition regardless of how the rest of the team
    # looks (>=2 overdue items, or any chronic-delay task) — protects against
    # everyone-is-drowning scenarios where the relative bar is too generous.
    for stats in per_assignee.values():
        relative_outlier = stats.overload_score > (mean_score + stdev_score) and stats.overload_score > 0
        absolute_red_flag = stats.overdue_count >= 2 or stats.chronic_delay_count >= 1
        stats.overload_flag = relative_outlier or absolute_red_flag

    ordered = sorted(per_assignee.values(), key=lambda s: s.overload_score, reverse=True)
    return WorkloadReport(as_of=as_of, stats=ordered, team_mean_score=mean_score, team_stdev_score=stdev_score)


def to_json(report: WorkloadReport) -> dict:
    return {
        "as_of": report.as_of.isoformat(),
        "team_mean_score": round(report.team_mean_score, 2),
        "team_stdev_score": round(report.team_stdev_score, 2),
        "assignees": [
            {
                "assignee_id": s.assignee_id,
                "name": s.assignee_name,
                "open": s.open_count,
                "overdue": s.overdue_count,
                "done": s.done_count,
                "blocked": s.blocked_count,
                "chronic_delay_tasks": s.chronic_delay_count,
                "due_within_window": s.due_within_window_count,
                "total_pushbacks": s.total_pushbacks,
                "overload_score": round(s.overload_score, 2),
                "overloaded": s.overload_flag,
            }
            for s in report.stats
        ],
    }


def format_report_text(report: WorkloadReport) -> str:
    header = f"LoopKeeper Workload Report — as of {report.as_of.isoformat()}"
    col = "{:<16}{:>6}{:>9}{:>7}{:>9}{:>9}{:>10}{:>10}"
    lines = [
        header,
        "=" * len(header),
        col.format("Owner", "Open", "Overdue", "Done", "DueSoon", "Chronic", "Pushbacks", "Score"),
        "-" * 84,
    ]
    for s in report.stats:
        flag = "  [OVERLOADED]" if s.overload_flag else ""
        lines.append(
            col.format(
                s.assignee_name[:15], s.open_count, s.overdue_count, s.done_count,
                s.due_within_window_count, s.chronic_delay_count, s.total_pushbacks,
                f"{s.overload_score:.1f}",
            ) + flag
        )
    lines.append("-" * 84)
    lines.append(f"Team baseline: mean={report.team_mean_score:.1f}, stdev={report.team_stdev_score:.1f}")
    return "\n".join(lines)
