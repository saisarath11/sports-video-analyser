"""
Unit tests for Player Tracker module.
"""

from core.tracker import PlayerTracker


def test_tracker_initialization():
    tracker = PlayerTracker()
    assert len(tracker.tracks) == 0
    assert len(tracker.get_active_player_ids()) == 0


def test_tracker_reset():
    tracker = PlayerTracker()
    tracker.reset()
    assert len(tracker.tracks) == 0
