const { ParkingSpot, Vehicle, ParkingSession } = require('../models/parking');

class ParkingController {
  // Get all parking spots
  static getAllSpots(req, res) {
    try {
      const spots = ParkingSpot.getAll();
      res.json({
        success: true,
        data: spots
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get spots by floor
  static getSpotsByFloor(req, res) {
    try {
      const floor = parseInt(req.params.floor);
      if (floor < 1 || floor > 5) {
        return res.status(400).json({
          success: false,
          message: 'Floor must be between 1 and 5'
        });
      }

      const spots = ParkingSpot.getByFloor(floor);
      res.json({
        success: true,
        data: spots
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get specific spot
  static getSpot(req, res) {
    try {
      const floor = parseInt(req.params.floor);
      const lot = parseInt(req.params.lot);

      if (floor < 1 || floor > 5 || lot < 1 || lot > 5) {
        return res.status(400).json({
          success: false,
          message: 'Floor and lot must be between 1 and 5'
        });
      }

      const spot = ParkingSpot.getByFloorAndLot(floor, lot);
      if (!spot) {
        return res.status(404).json({
          success: false,
          message: 'Spot not found'
        });
      }

      res.json({
        success: true,
        data: spot
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get available spots
  static getAvailableSpots(req, res) {
    try {
      const spots = ParkingSpot.getAvailable();
      res.json({
        success: true,
        count: spots.length,
        data: spots
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get occupied spots
  static getOccupiedSpots(req, res) {
    try {
      const spots = ParkingSpot.getOccupied();
      res.json({
        success: true,
        count: spots.length,
        data: spots
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get occupancy statistics
  static getOccupancyStats(req, res) {
    try {
      const stats = ParkingSpot.getOccupancyStats();
      res.json({
        success: true,
        data: stats
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Check in a vehicle (camera detection)
  static checkIn(req, res) {
    try {
      const { vehiclePlate, floor, lot } = req.body;

      if (!vehiclePlate || !floor || !lot) {
        return res.status(400).json({
          success: false,
          message: 'vehiclePlate, floor, and lot are required'
        });
      }

      if (floor < 1 || floor > 5 || lot < 1 || lot > 5) {
        return res.status(400).json({
          success: false,
          message: 'Floor and lot must be between 1 and 5'
        });
      }

      const session = ParkingSession.create(vehiclePlate, floor, lot);
      res.json({
        success: true,
        message: `Vehicle ${vehiclePlate} checked in to Floor ${floor}, Lot ${lot}`,
        data: session
      });
    } catch (error) {
      res.status(400).json({
        success: false,
        message: error.message
      });
    }
  }

  // Check out a vehicle
  static checkOut(req, res) {
    try {
      const { vehiclePlate } = req.body;

      if (!vehiclePlate) {
        return res.status(400).json({
          success: false,
          message: 'vehiclePlate is required'
        });
      }

      const session = ParkingSession.checkout(vehiclePlate);
      res.json({
        success: true,
        message: `Vehicle ${vehiclePlate} checked out`,
        data: session
      });
    } catch (error) {
      res.status(400).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get active parking sessions
  static getActiveSessions(req, res) {
    try {
      const sessions = ParkingSession.getActive();
      res.json({
        success: true,
        count: sessions.length,
        data: sessions
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get all vehicles
  static getAllVehicles(req, res) {
    try {
      const vehicles = Vehicle.getAll();
      res.json({
        success: true,
        count: vehicles.length,
        data: vehicles
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get vehicle by plate
  static getVehicle(req, res) {
    try {
      const { plate } = req.params;
      const vehicle = Vehicle.getByPlate(plate);

      if (!vehicle) {
        return res.status(404).json({
          success: false,
          message: 'Vehicle not found'
        });
      }

      const sessions = ParkingSession.getByVehicle(plate);
      res.json({
        success: true,
        data: {
          vehicle,
          sessions
        }
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }
}

module.exports = ParkingController;
