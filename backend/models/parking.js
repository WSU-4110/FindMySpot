const parkingSpots = [];
const parkingSessions = [];
const vehicles = [];

// Initialize parking spots for floors 1-5 and lots 1-5
function initializeParkingSpots() {
  const spots = [];
  for (let floor = 1; floor <= 5; floor++) {
    for (let lot = 1; lot <= 5; lot++) {
      spots.push({
        id: spots.length + 1,
        floor: floor,
        lot: lot,
        occupied: false,
        vehicle: null,
        checkInTime: null
      });
    }
  }
  return spots;
}

// Initialize the parking spots array
parkingSpots.push(...initializeParkingSpots());

class ParkingSpot {
  static getAll() {
    return parkingSpots;
  }

  static getByFloorAndLot(floor, lot) {
    return parkingSpots.find(spot => spot.floor === floor && spot.lot === lot);
  }

  static getByFloor(floor) {
    return parkingSpots.filter(spot => spot.floor === floor);
  }

  static getAvailable() {
    return parkingSpots.filter(spot => !spot.occupied);
  }

  static getOccupied() {
    return parkingSpots.filter(spot => spot.occupied);
  }

  static updateOccupancy(floor, lot, occupied, vehiclePlate = null) {
    const spot = this.getByFloorAndLot(floor, lot);
    if (spot) {
      spot.occupied = occupied;
      spot.vehicle = vehiclePlate;
      spot.checkInTime = occupied ? new Date().toISOString() : null;
      return spot;
    }
    return null;
  }

  static getOccupancyStats() {
    const total = parkingSpots.length;
    const occupied = parkingSpots.filter(spot => spot.occupied).length;
    const available = total - occupied;
    
    const byFloor = {};
    for (let floor = 1; floor <= 5; floor++) {
      const floorSpots = parkingSpots.filter(spot => spot.floor === floor);
      const floorOccupied = floorSpots.filter(spot => spot.occupied).length;
      byFloor[floor] = {
        total: floorSpots.length,
        occupied: floorOccupied,
        available: floorSpots.length - floorOccupied
      };
    }

    return {
      total,
      occupied,
      available,
      byFloor
    };
  }
}

class Vehicle {
  static create(plate) {
    const existing = vehicles.find(v => v.plate === plate);
    if (existing) {
      return existing;
    }
    
    const vehicle = {
      id: vehicles.length + 1,
      plate: plate,
      createdAt: new Date().toISOString()
    };
    vehicles.push(vehicle);
    return vehicle;
  }

  static getByPlate(plate) {
    return vehicles.find(v => v.plate === plate);
  }

  static getAll() {
    return vehicles;
  }
}

class ParkingSession {
  static create(vehiclePlate, floor, lot) {
    const vehicle = Vehicle.create(vehiclePlate);
    const spot = ParkingSpot.getByFloorAndLot(floor, lot);
    
    if (!spot) {
      throw new Error(`Spot not found: Floor ${floor}, Lot ${lot}`);
    }

    if (spot.occupied) {
      throw new Error(`Spot already occupied: Floor ${floor}, Lot ${lot}`);
    }

    const session = {
      id: parkingSessions.length + 1,
      vehicleId: vehicle.id,
      vehiclePlate: vehicle.plate,
      spotId: spot.id,
      floor: floor,
      lot: lot,
      checkInTime: new Date().toISOString(),
      checkOutTime: null
    };

    parkingSessions.push(session);
    ParkingSpot.updateOccupancy(floor, lot, true, vehiclePlate);

    return session;
  }

  static checkout(vehiclePlate) {
    const activeSession = parkingSessions.find(
      s => s.vehiclePlate === vehiclePlate && s.checkOutTime === null
    );

    if (!activeSession) {
      throw new Error(`No active session found for vehicle: ${vehiclePlate}`);
    }

    activeSession.checkOutTime = new Date().toISOString();
    ParkingSpot.updateOccupancy(activeSession.floor, activeSession.lot, false, null);

    return activeSession;
  }

  static getActive() {
    return parkingSessions.filter(s => s.checkOutTime === null);
  }

  static getByVehicle(vehiclePlate) {
    return parkingSessions.filter(s => s.vehiclePlate === vehiclePlate);
  }

  static getAll() {
    return parkingSessions;
  }
}

module.exports = {
  ParkingSpot,
  Vehicle,
  ParkingSession
};
