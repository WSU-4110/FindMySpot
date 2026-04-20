const { ParkingSpot, Vehicle, ParkingSession, SecurityFlag } = require('../models/parking');
const { User } = require('../models/user');
const UserVehicle = require('../models/vehicle');
const Notification = require('../models/notification');
const DetectionEvent = require('../models/detection');

// Track auto-checkout timers (in-memory)
const autoCheckoutTimers = {};
let lastKnownOccupancyStats = null;

// Helper to schedule auto-checkout after 10 minutes
function scheduleAutoCheckout(licensePlate, delayMs = 10 * 60 * 1000) {
  // Cancel any existing timer for this plate
  if (autoCheckoutTimers[licensePlate]) {
    clearTimeout(autoCheckoutTimers[licensePlate]);
  }

  // Schedule new timer
  autoCheckoutTimers[licensePlate] = setTimeout(async () => {
    try {
      await ParkingSession.checkout(licensePlate);
      console.log(`[AUTO-CHECKOUT] Vehicle ${licensePlate} auto-checked out after 10 minutes`);
    } catch (error) {
      console.error(`[AUTO-CHECKOUT] Failed to auto-checkout ${licensePlate}: ${error.message}`);
      try {
        const session = await ParkingSession.getActiveByVehicle(licensePlate);
        if (session) {
          await ParkingSpot.updateOccupancy(session.floor, session.lot, false, null);
          console.log(`[AUTO-CHECKOUT] Force-freed spot for ${licensePlate} after failed checkout`);
        }
      } catch (occupancyError) {
        console.error(`[AUTO-CHECKOUT] Also failed to free spot: ${occupancyError.message}`);
      }
    }
    delete autoCheckoutTimers[licensePlate];
  }, delayMs);

  console.log(`[AUTO-CHECKOUT] Scheduled auto-checkout for ${licensePlate} in 10 minutes`);
}

class ParkingController {
  // Get all parking spots
  static async getAllSpots(req, res) {
    try {
      const spots = await ParkingSpot.getAll();
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
  static async getSpotsByFloor(req, res) {
    try {
      const floor = parseInt(req.params.floor);
      if (floor < 1 || floor > 5) {
        return res.status(400).json({
          success: false,
          message: 'Floor must be between 1 and 5'
        });
      }

      const spots = await ParkingSpot.getByFloor(floor);
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
  static async getSpot(req, res) {
    try {
      const floor = parseInt(req.params.floor);
      const lot = parseInt(req.params.lot);

      if (floor < 1 || floor > 5 || lot < 1 || lot > 5) {
        return res.status(400).json({
          success: false,
          message: 'Floor and lot must be between 1 and 5'
        });
      }

      const spot = await ParkingSpot.getByFloorAndLot(floor, lot);
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
  static async getAvailableSpots(req, res) {
    try {
      const spots = await ParkingSpot.getAvailable();
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
  static async getOccupiedSpots(req, res) {
    try {
      const spots = await ParkingSpot.getOccupied();
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
  static async getOccupancyStats(req, res) {
    try {
      const stats = await ParkingSpot.getOccupancyStats();
      lastKnownOccupancyStats = stats;
      res.json({
        success: true,
        stale: false,
        data: stats
      });
    } catch (error) {
      if (lastKnownOccupancyStats) {
        return res.json({
          success: true,
          stale: true,
          message: 'Real-time update unavailable. Showing the most recent known occupancy data.',
          data: lastKnownOccupancyStats
        });
      }

      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Check in a vehicle (camera detection)
  static async checkIn(req, res) {
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

      const normalizedPlate = String(vehiclePlate || '').toUpperCase().trim();
      const locationStr = `Floor ${floor}, Lot ${lot}`;
      const session = await ParkingSession.create(normalizedPlate, floor, lot);
      
      // Schedule auto-checkout after 10 minutes
      scheduleAutoCheckout(normalizedPlate);

      // Mirror camera-detection behavior so manual check-ins also generate notifications.
      let detection = null;
      const notifications = [];
      const notificationErrors = [];

      try {
        detection = await DetectionEvent.recordDetection(
          normalizedPlate,
          floor,
          lot,
          locationStr,
          0.98,
          'MANUAL_CHECKIN',
          null,
          null
        );
      } catch (error) {
        notificationErrors.push({
          step: 'recordDetection',
          error: error.message
        });
      }

      try {
        const matchingVehicles = await UserVehicle.getAllByLicensePlate(normalizedPlate);
        for (const vehicle of matchingVehicles) {
          try {
            const notification = await Notification.create(
              vehicle.user_id,
              vehicle.id,
              detection?.id || null,
              'Vehicle Detected',
              `Your registered license plate ${normalizedPlate} was detected at ${locationStr}`,
              locationStr,
              detection?.detected_at || new Date().toISOString()
            );
            notifications.push(notification);
          } catch (error) {
            notificationErrors.push({
              userId: vehicle.user_id,
              error: error.message
            });
          }
        }
      } catch (error) {
        notificationErrors.push({
          step: 'loadMatchingVehicles',
          error: error.message
        });
      }
      
      res.json({
        success: true,
        message: `Vehicle ${normalizedPlate} checked in to Floor ${floor}, Lot ${lot}`,
        data: session,
        notificationSummary: {
          created: notifications.length,
          errors: notificationErrors
        }
      });
    } catch (error) {
      res.status(400).json({
        success: false,
        message: error.message
      });
    }
  }

  // Check out a vehicle
  static async checkOut(req, res) {
    try {
      const { vehiclePlate } = req.body;

      if (!vehiclePlate) {
        return res.status(400).json({
          success: false,
          message: 'vehiclePlate is required'
        });
      }

      const session = await ParkingSession.checkout(vehiclePlate);

      if (autoCheckoutTimers[vehiclePlate]) {
        clearTimeout(autoCheckoutTimers[vehiclePlate]);
        delete autoCheckoutTimers[vehiclePlate];
      }

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
  static async getActiveSessions(req, res) {
    try {
      const sessions = await ParkingSession.getActive();
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
  static async getAllVehicles(req, res) {
    try {
      const vehicles = await Vehicle.getAll();
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
  static async getVehicle(req, res) {
    try {
      const { plate } = req.params;
      const vehicle = await Vehicle.getByPlate(plate);

      if (!vehicle) {
        return res.status(404).json({
          success: false,
          message: 'Vehicle not found'
        });
      }

      const sessions = await ParkingSession.getByVehicle(plate);
      const location = await ParkingSession.locateVehicle(plate);
      res.json({
        success: true,
        data: {
          vehicle,
          sessions,
          location
        }
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // FR5 - Precise mapping details for a located vehicle
  static async locateVehicleWithSpot(req, res) {
    try {
      const { plate } = req.params;
      const location = await ParkingSession.locateVehicle(plate);

      if (!location) {
        return res.status(404).json({
          success: false,
          message: 'Vehicle location not found'
        });
      }

      const hasSpotData = location.floor != null && location.lot != null;
      res.json({
        success: true,
        preciseSpotAvailable: hasSpotData,
        message: hasSpotData
          ? 'Exact parking spot found.'
          : 'Spot data is missing. Showing available location details only.',
        data: {
          vehiclePlate: location.vehiclePlate,
          floor: location.floor,
          area: location.area,
          lot: location.lot,
          spotNumber: location.spotNumber,
          locationDescription: location.locationDescription,
          parkedSince: location.checkInTime,
          sessionActive: location.sessionActive
        }
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // FR7 - Return visual or step-by-step directions
  static async getVehicleDirections(req, res) {
    try {
      const { plate } = req.params;
      const directions = await ParkingSession.getDirectionsForVehicle(plate);

      if (!directions) {
        return res.status(404).json({
          success: false,
          message: 'Vehicle location not found for directions.'
        });
      }

      if (!directions.navigationAvailable) {
        return res.json({
          success: true,
          navigationAvailable: false,
          message: 'Navigation data unavailable. Displaying static location details.',
          data: directions
        });
      }

      res.json({
        success: true,
        navigationAvailable: true,
        data: directions
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // FR11 - Usage and efficiency analytics reporting
  static async getUsageAnalyticsReport(req, res) {
    try {
      const { hoursBack = 24 } = req.query;
      const report = await ParkingSession.getUsageReport(hoursBack);
      res.json({
        success: true,
        data: report
      });
    } catch (error) {
      console.error('[ANALYTICS] Failed to generate report:', error.message);
      res.status(500).json({
        success: false,
        message: 'Report generation failed. Administrators have been notified.'
      });
    }
  }

  // FR12 - Scan and persist security flags
  static async runSecurityFlagScan(req, res) {
  try {
    const { token } = req.params;

    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Token is required'
      });
    }

    const user = await User.getByToken(token);

    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }

    const { maxDurationHours = 24 } = req.query;
    const result = await ParkingSession.scanAndFlagSecurityIssues(maxDurationHours);

    res.json({
      success: true,
      message: 'Security flag scan completed.',
      data: result
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
}

  static async getOpenSecurityFlags(req, res) {
    try {
      const { limit = 100 } = req.query;
      const flags = await SecurityFlag.getOpen(limit);
      res.json({
        success: true,
        count: flags.length,
        data: flags
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
