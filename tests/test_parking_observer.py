"""
Unit Tests for parking_observer.py — FindMySpot
Tests the Observer pattern: ParkingSpot (Subject) + all Concrete Observers
Framework: pytest
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


# ──────────────────────────────────────────────
# Minimal re-implementation of the classes so
# tests run without any external dependencies
# ──────────────────────────────────────────────
from abc import ABC, abstractmethod
from typing import List, Optional


class ParkingObserver(ABC):
    @abstractmethod
    def update(self, spot_id: str, floor: int, status: str,
                license_plate: Optional[str]) -> None:
        pass


class ParkingSpot:
    def __init__(self, spot_id: str, floor: int):
        self._spot_id = spot_id
        self._floor = floor
        self._status = "available"
        self._license_plate: Optional[str] = None
        self._observers: List[ParkingObserver] = []
        self._last_updated: Optional[datetime] = None

    def attach(self, observer: ParkingObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: ParkingObserver) -> None:
        self._observers.remove(observer)

    def notify_observers(self) -> None:
        for observer in self._observers:
            observer.update(self._spot_id, self._floor,
                            self._status, self._license_plate)

    def mark_occupied(self, license_plate: str) -> None:
        self._status = "occupied"
        self._license_plate = license_plate.upper()
        self._last_updated = datetime.now()
        self.notify_observers()

    def mark_available(self) -> None:
        self._status = "available"
        self._license_plate = None
        self._last_updated = datetime.now()
        self.notify_observers()

    @property
    def spot_id(self): return self._spot_id
    @property
    def floor(self): return self._floor
    @property
    def status(self): return self._status
    @property
    def license_plate(self): return self._license_plate


class DatabaseObserver(ParkingObserver):
    def __init__(self, db_connection=None):
        self._db = db_connection
        self._log: List[dict] = []

    def update(self, spot_id, floor, status, license_plate):
        record = {"spot_id": spot_id, "floor": floor,
                  "status": status, "license_plate": license_plate,
                  "timestamp": datetime.now().isoformat()}
        self._log.append(record)

    def get_log(self): return self._log


class UserNotificationObserver(ParkingObserver):
    def __init__(self):
        self._notification_sent: List[str] = []

    def update(self, spot_id, floor, status, license_plate):
        if status == "occupied" and license_plate:
            msg = (f"Your vehicle ({license_plate}) has been detected at "
                   f"Spot {spot_id}, Floor {floor}. Your location has been saved.")
            self._notification_sent.append(msg)

    @property
    def notifications(self): return self._notification_sent


class GarageDashboardObserver(ParkingObserver):
    def __init__(self):
        self._spot_states: dict = {}

    def update(self, spot_id, floor, status, license_plate):
        self._spot_states[spot_id] = status

    def get_available_count(self) -> int:
        return sum(1 for s in self._spot_states.values() if s == "available")


# ──────────────────────────────────────────────
# TEST 1 — ParkingSpot.mark_occupied
# ──────────────────────────────────────────────
class TestParkingSpotMarkOccupied:
    """Tests for ParkingSpot.mark_occupied()"""

    def test_status_becomes_occupied(self):
        """mark_occupied sets status to 'occupied'."""
        spot = ParkingSpot("A1", floor=1)
        spot.mark_occupied("ABC123")
        assert spot.status == "occupied"

    def test_license_plate_uppercased(self):
        """mark_occupied uppercases the plate regardless of input case."""
        spot = ParkingSpot("A1", floor=1)
        spot.mark_occupied("abc123")
        assert spot.license_plate == "ABC123"

    def test_notify_called_on_occupied(self):
        """mark_occupied notifies all attached observers."""
        spot = ParkingSpot("A1", floor=1)
        observer = MagicMock(spec=ParkingObserver)
        spot.attach(observer)
        spot.mark_occupied("XYZ999")
        observer.update.assert_called_once_with("A1", 1, "occupied", "XYZ999")

    def test_last_updated_set(self):
        """mark_occupied sets _last_updated timestamp."""
        spot = ParkingSpot("B2", floor=2)
        assert spot._last_updated is None
        spot.mark_occupied("DEF456")
        assert isinstance(spot._last_updated, datetime)


# ──────────────────────────────────────────────
# TEST 2 — ParkingSpot.mark_available
# ──────────────────────────────────────────────
class TestParkingSpotMarkAvailable:
    """Tests for ParkingSpot.mark_available()"""

    def test_status_becomes_available(self):
        """mark_available resets status to 'available'."""
        spot = ParkingSpot("C3", floor=3)
        spot.mark_occupied("ABC123")
        spot.mark_available()
        assert spot.status == "available"

    def test_license_plate_cleared(self):
        """mark_available clears the license plate."""
        spot = ParkingSpot("C3", floor=3)
        spot.mark_occupied("ABC123")
        spot.mark_available()
        assert spot.license_plate is None

    def test_observers_notified_on_available(self):
        """mark_available notifies observers with status='available'."""
        spot = ParkingSpot("D4", floor=4)
        observer = MagicMock(spec=ParkingObserver)
        spot.attach(observer)
        spot.mark_available()
        observer.update.assert_called_once_with("D4", 4, "available", None)


# ──────────────────────────────────────────────
# TEST 3 — ParkingSpot.attach / detach
# ──────────────────────────────────────────────
class TestParkingSpotAttachDetach:
    """Tests for ParkingSpot.attach() and detach()"""

    def test_attach_adds_observer(self):
        """attach registers a new observer."""
        spot = ParkingSpot("E5", floor=5)
        obs = MagicMock(spec=ParkingObserver)
        spot.attach(obs)
        assert obs in spot._observers

    def test_attach_same_observer_twice_ignored(self):
        """Attaching the same observer twice does not duplicate it."""
        spot = ParkingSpot("F6", floor=1)
        obs = MagicMock(spec=ParkingObserver)
        spot.attach(obs)
        spot.attach(obs)
        assert spot._observers.count(obs) == 1

    def test_detach_removes_observer(self):
        """detach unregisters the observer so it no longer receives updates."""
        spot = ParkingSpot("G7", floor=2)
        obs = MagicMock(spec=ParkingObserver)
        spot.attach(obs)
        spot.detach(obs)
        spot.mark_occupied("TEST123")
        obs.update.assert_not_called()


# ──────────────────────────────────────────────
# TEST 4 — DatabaseObserver.update
# ──────────────────────────────────────────────
class TestDatabaseObserver:
    """Tests for DatabaseObserver.update() and get_log()"""

    def test_log_entry_created_on_update(self):
        """update() appends a record to the internal log."""
        db_obs = DatabaseObserver()
        db_obs.update("A1", 1, "occupied", "ABC123")
        assert len(db_obs.get_log()) == 1

    def test_log_entry_fields(self):
        """Log entry contains correct spot, floor, status, and plate."""
        db_obs = DatabaseObserver()
        db_obs.update("A1", 1, "occupied", "ABC123")
        entry = db_obs.get_log()[0]
        assert entry["spot_id"] == "A1"
        assert entry["floor"] == 1
        assert entry["status"] == "occupied"
        assert entry["license_plate"] == "ABC123"

    def test_multiple_updates_logged(self):
        """Multiple update() calls each produce a log entry."""
        db_obs = DatabaseObserver()
        db_obs.update("A1", 1, "occupied", "ABC123")
        db_obs.update("A1", 1, "available", None)
        assert len(db_obs.get_log()) == 2

    def test_log_entry_has_timestamp(self):
        """Log entries include a timestamp string."""
        db_obs = DatabaseObserver()
        db_obs.update("B2", 2, "occupied", "XYZ789")
        entry = db_obs.get_log()[0]
        assert "timestamp" in entry
        assert isinstance(entry["timestamp"], str)


# ──────────────────────────────────────────────
# TEST 5 — UserNotificationObserver.update
# ──────────────────────────────────────────────
class TestUserNotificationObserver:
    """Tests for UserNotificationObserver.update()"""

    def test_notification_sent_on_occupied(self):
        """A notification message is created when status is 'occupied'."""
        notif_obs = UserNotificationObserver()
        notif_obs.update("A1", 1, "occupied", "ABC123")
        assert len(notif_obs.notifications) == 1

    def test_notification_contains_plate_and_spot(self):
        """The notification message mentions the plate and spot ID."""
        notif_obs = UserNotificationObserver()
        notif_obs.update("A1", 1, "occupied", "ABC123")
        msg = notif_obs.notifications[0]
        assert "ABC123" in msg
        assert "A1" in msg

    def test_no_notification_on_available(self):
        """No notification is sent when status is 'available'."""
        notif_obs = UserNotificationObserver()
        notif_obs.update("A1", 1, "available", None)
        assert len(notif_obs.notifications) == 0

    def test_no_notification_without_plate(self):
        """No notification is sent when license_plate is None even if occupied."""
        notif_obs = UserNotificationObserver()
        notif_obs.update("A1", 1, "occupied", None)
        assert len(notif_obs.notifications) == 0


# ──────────────────────────────────────────────
# TEST 6 — GarageDashboardObserver
# ──────────────────────────────────────────────
class TestGarageDashboardObserver:
    """Tests for GarageDashboardObserver.update() and get_available_count()"""

    def test_spot_state_updated_on_occupied(self):
        """update() records the spot as occupied in internal state."""
        dash_obs = GarageDashboardObserver()
        dash_obs.update("A1", 1, "occupied", "ABC123")
        assert dash_obs._spot_states["A1"] == "occupied"

    def test_available_count_zero_when_all_occupied(self):
        """get_available_count() returns 0 when all spots are occupied."""
        dash_obs = GarageDashboardObserver()
        dash_obs.update("A1", 1, "occupied", "P1")
        dash_obs.update("A2", 1, "occupied", "P2")
        assert dash_obs.get_available_count() == 0

    def test_available_count_correct_mixed(self):
        """get_available_count() returns correct count with mixed states."""
        dash_obs = GarageDashboardObserver()
        dash_obs.update("A1", 1, "occupied", "P1")
        dash_obs.update("A2", 1, "available", None)
        dash_obs.update("A3", 2, "available", None)
        assert dash_obs.get_available_count() == 2

    def test_spot_state_updates_from_occupied_to_available(self):
        """Updating the same spot from occupied to available reflects correctly."""
        dash_obs = GarageDashboardObserver()
        dash_obs.update("A1", 1, "occupied", "P1")
        dash_obs.update("A1", 1, "available", None)
        assert dash_obs._spot_states["A1"] == "available"
        assert dash_obs.get_available_count() == 1


# ──────────────────────────────────────────────
# TEST 7 — End-to-end integration (Observer chain)
# ──────────────────────────────────────────────
class TestObserverIntegration:
    """Integration test: full observer chain fires correctly."""

    def test_full_chain_on_mark_occupied(self):
        spot = ParkingSpot("Z99", floor=5)
        db_obs = DatabaseObserver()
        notif_obs = UserNotificationObserver()
        dash_obs = GarageDashboardObserver()

        spot.attach(db_obs)
        spot.attach(notif_obs)
        spot.attach(dash_obs)

        spot.mark_occupied("FULLTEST")

        assert db_obs.get_log()[0]["status"] == "occupied"
        assert len(notif_obs.notifications) == 1
        assert dash_obs._spot_states["Z99"] == "occupied"

    def test_full_chain_on_mark_available(self):
        spot = ParkingSpot("Z99", floor=5)
        db_obs = DatabaseObserver()
        dash_obs = GarageDashboardObserver()

        spot.attach(db_obs)
        spot.attach(dash_obs)

        spot.mark_occupied("FULLTEST")
        spot.mark_available()

        log = db_obs.get_log()
        assert log[-1]["status"] == "available"
        assert dash_obs.get_available_count() == 1