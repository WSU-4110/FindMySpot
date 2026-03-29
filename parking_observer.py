"""
FindMySpot - Observer Design Pattern Implementation
Branch: your_name_homework4
Commit message: "Apply Observer pattern to parking spot status notification system"

Pattern: Observer (also known as Publish-Subscribe)
Applied to: Parking spot status change events
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional




class ParkingObserver(ABC):
    """Abstract base class for all parking observers."""

    @abstractmethod
    def update(self, spot_id: str, floor: int, status: str, license_plate: Optional[str]) -> None:
        """Called by the subject when parking spot status changes."""
        pass




class ParkingSpot:
    """
    Subject in the Observer pattern.
    Represents a single parking spot that notifies observers on status changes.
    """

    def __init__(self, spot_id: str, floor: int):
        self._spot_id = spot_id
        self._floor = floor
        self._status = "available"           # "available" | "occupied"
        self._license_plate: Optional[str] = None
        self._observers: List[ParkingObserver] = []
        self._last_updated: Optional[datetime] = None


    def attach(self, observer: ParkingObserver) -> None:
        """Register an observer to receive status updates."""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: ParkingObserver) -> None:
        """Unregister an observer."""
        self._observers.remove(observer)

    def notify_observers(self) -> None:
        """Notify all registered observers of the current state."""
        for observer in self._observers:
            observer.update(
                self._spot_id,
                self._floor,
                self._status,
                self._license_plate
            )


    def mark_occupied(self, license_plate: str) -> None:
        """ANPR camera calls this when a car parks in the spot."""
        self._status = "occupied"
        self._license_plate = license_plate.upper()
        self._last_updated = datetime.now()
        self.notify_observers()

    def mark_available(self) -> None:
        """ANPR camera calls this when the car leaves."""
        self._status = "available"
        self._license_plate = None
        self._last_updated = datetime.now()
        self.notify_observers()


    @property
    def spot_id(self) -> str:
        return self._spot_id

    @property
    def floor(self) -> int:
        return self._floor

    @property
    def status(self) -> str:
        return self._status

    @property
    def license_plate(self) -> Optional[str]:
        return self._license_plate




class DatabaseObserver(ParkingObserver):
    """
    Concrete Observer 1: Persists spot status to the database.
    In production this would call the Node.js/Express backend API.
    """

    def __init__(self, db_connection=None):
        self._db = db_connection  # Placeholder for actual DB connection
        self._log: List[dict] = []  # In-memory log for demo

    def update(self, spot_id: str, floor: int, status: str, license_plate: Optional[str]) -> None:
        record = {
            "spot_id": spot_id,
            "floor": floor,
            "status": status,
            "license_plate": license_plate,
            "timestamp": datetime.now().isoformat()
        }
        self._log.append(record)
        # In production: self._db.execute("UPDATE spots SET status=... WHERE spot_id=...")
        print(f"[DatabaseObserver]  Saved → spot={spot_id}, floor={floor}, "
              f"status={status}, plate={license_plate}")

    def get_log(self) -> List[dict]:
        return self._log


class UserNotificationObserver(ParkingObserver):
    """
    Concrete Observer 2: Sends a push notification to the user
    whose license plate just parked or whose car was located.
    """

    def __init__(self):
        self._notification_sent: List[str] = []

    def update(self, spot_id: str, floor: int, status: str, license_plate: Optional[str]) -> None:
        if status == "occupied" and license_plate:
            message = (
                f"Your vehicle ({license_plate}) has been detected at "
                f"Spot {spot_id}, Floor {floor}. Your location has been saved."
            )
            self._notification_sent.append(message)
            # In production: call Firebase FCM / APNs push notification API
            print(f"[UserNotificationObserver]  Push → {message}")

        elif status == "available":
            print(f"[UserNotificationObserver]  Spot {spot_id} is now available.")


class GarageDashboardObserver(ParkingObserver):
    """
    Concrete Observer 3: Updates the real-time garage availability 
    dashboard (the red/green indicator lights and the kiosk display).
    """

    def __init__(self):
        self._spot_states: dict = {}

    def update(self, spot_id: str, floor: int, status: str, license_plate: Optional[str]) -> None:
        self._spot_states[spot_id] = status
        color = " OCCUPIED" if status == "occupied" else " AVAILABLE"
        print(f"[GarageDashboardObserver]  Display updated → Spot {spot_id} "
              f"(Floor {floor}): {color}")

    def get_available_count(self) -> int:
        return sum(1 for s in self._spot_states.values() if s == "available")



if __name__ == "__main__":
    print("=" * 60)
    print("  FindMySpot — Observer Pattern Demo")
    print("=" * 60)

    # Create parking spots (subjects)
    spot_A12 = ParkingSpot(spot_id="A12", floor=2)
    spot_B07 = ParkingSpot(spot_id="B07", floor=3)

    # Create observers
    db_observer     = DatabaseObserver()
    notif_observer  = UserNotificationObserver()
    dash_observer   = GarageDashboardObserver()

    # Register observers with spots
    spot_A12.attach(db_observer)
    spot_A12.attach(notif_observer)
    spot_A12.attach(dash_observer)

    spot_B07.attach(db_observer)
    spot_B07.attach(dash_observer)   # B07 doesn't send user push notifs in this example

    print("\n--- Car parks in Spot A12 (ANPR detects plate ABC-1234) ---")
    spot_A12.mark_occupied("ABC-1234")

    print("\n--- Another car parks in Spot B07 ---")
    spot_B07.mark_occupied("XYZ-9999")

    print("\n--- Car leaves Spot A12 ---")
    spot_A12.mark_available()

    print(f"\n--- Available spots on dashboard: {dash_observer.get_available_count()} ---")
    print("\n--- Database log ---")
    for entry in db_observer.get_log():
        print(f"  {entry}")