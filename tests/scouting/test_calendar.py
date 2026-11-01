from datetime import datetime, timezone
import pytest

def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

@pytest.mark.parametrize("start,expected", [
    ("2026-11-06T14:17:00Z", "2026-11-09T14:17:00Z"),
    ("2026-10-30T13:17:00Z", "2026-11-02T14:17:00Z"),
    ("2026-03-06T14:17:00Z", "2026-03-09T13:17:00Z"),
    ("2026-11-05T20:00:00Z", "2026-11-06T14:17:00Z"),
    ("2026-11-02T14:16:00Z", "2026-11-02T14:17:00Z"),
])
def test_next_primary_uses_local_weekdays_and_dst(start, expected):
    from wbb_scouting.calendar import next_primary_check
    assert next_primary_check(utc(start)) == utc(expected)

def test_window_covers_weekend_without_missing_run_warning():
    from wbb_scouting.calendar import refresh_window, schedule_health
    now = utc("2026-10-30T13:17:00Z")
    assert refresh_window(now) == (now, utc("2026-11-02T14:17:00Z"))
    assert schedule_health(now, utc("2026-11-01T20:00:00Z")) == "on_schedule"
    assert schedule_health(now, utc("2026-11-02T17:00:00Z")) == "overdue"

@pytest.mark.parametrize("status,pending,expected", [
    ("complete", False, False), ("partial", True, True), ("failed", False, True),
])
def test_retry_uses_primary_outcome(status, pending, expected):
    from wbb_scouting.calendar import retry_required
    primary = {"attempted_at": "2026-11-02T14:17:00Z", "status": status, "pending": pending}
    assert retry_required(utc("2026-11-02T16:17:00Z"), primary) is expected

def test_retry_recovers_missing_primary_and_skips_off_days():
    from wbb_scouting.calendar import retry_required
    assert retry_required(utc("2026-11-02T16:17:00Z"), None)
    assert not retry_required(utc("2026-11-03T16:17:00Z"), None)

def test_naive_clock_is_rejected():
    from wbb_scouting.calendar import next_primary_check
    with pytest.raises(ValueError, match="timezone"):
        next_primary_check(datetime(2026, 11, 2))
