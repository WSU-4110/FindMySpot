const DetectionEvent = require('../models/detection');
const UserVehicle = require('../models/vehicle');
const Notification = require('../models/notification');
const { User } = require('../models/user');
const { ParkingSession, Vehicle, ParkingSpot } = require('../models/parking');
const fs = require('fs');
const path = require('path');

// Track auto-checkout timers (in-memory)
const autoCheckoutTimers = {};

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

function clearAutoCheckoutTimer(licensePlate) {
  if (autoCheckoutTimers[licensePlate]) {
    clearTimeout(autoCheckoutTimers[licensePlate]);
    delete autoCheckoutTimers[licensePlate];
  }
}

function normalizeText(value) {
  return String(value || '').trim().toUpperCase();
}

function getCameraConfig(cameraId) {
  try {
    const configPath = path.join(__dirname, '../../camera_config.json');
    const raw = fs.readFileSync(configPath, 'utf8');
    const parsed = JSON.parse(raw);
    const cameras = Array.isArray(parsed.cameras) ? parsed.cameras : [];

    return cameras.find((camera) => {
      return String(camera.camera_id) === String(cameraId);
    }) || null;
  } catch (error) {
    console.warn(`[CONFIG] Failed to load camera_config.json: ${error.message}`);
    return null;
  }
}

function isExitEvent(eventType, location, cameraId) {
  const normalizedEventType = normalizeText(eventType);

  if (normalizedEventType === 'EXIT') return true;
  if (normalizedEventType === 'ENTRY') return false;

  const cameraConfig = getCameraConfig(cameraId);
  if (cameraConfig) {
    if (cameraConfig.is_exit_camera === true) return true;
    if (cameraConfig.is_entry_camera === true) return false;
  }

  const locationText = normalizeText(location);
  return locationText.includes('GATE OUT') || locationText.includes('EXIT GATE');
}

class DetectionController {
  // Record a detected license plate (called by cameras/AI service)
  static async recordPlateDetection(req, res) {
    try {
      const { 
        licensePlate, 
        floor, 
        lot, 
        location, 
        confidence = 0.95,
        cameraId,
        latitude,
        longitude,
        eventType = 'ENTRY'
      } = req.body;

      if (!licensePlate) {
        return res.status(400).json({
          success: false,
          message: 'License plate is required'
        });
      }

      // Record the detection
      const normalizedPlate = String(licensePlate || '').toUpperCase().trim();
      const exitEvent = isExitEvent(eventType, location, cameraId);

      const detection = await DetectionEvent.recordDetection(
        normalizedPlate,
        floor,
        lot,
        location,
        confidence,
        cameraId,
        latitude,
        longitude
      );

      // Create parking session if floor and lot are provided
      let parkingSession = null;
      let parkingError = null;
      let sessionClosedByExit = null;
      if (exitEvent) {
        try {
          sessionClosedByExit = await ParkingSession.closeByExitDetection(normalizedPlate, {
            cameraId: cameraId || null,
            detectedAt: detection.detected_at,
            location: location || null
          });
          clearAutoCheckoutTimer(normalizedPlate);
          console.log(`[PARKING] Exit detected. Session closed for ${normalizedPlate}`);
        } catch (err) {
          parkingError = `Exit detected but no active session was closed: ${err.message}`;
          console.error(`[PARKING] ${parkingError}`);
        }
      } else if (floor != null && lot != null) {
        try {
          // Create vehicle record
          await Vehicle.create(normalizedPlate);
          
          // Create or update parking session
          parkingSession = await ParkingSession.create(normalizedPlate, floor, lot);
          console.log(`[PARKING] Session created for ${normalizedPlate} at Floor ${floor}, Lot ${lot}`);
          
          // Schedule auto-checkout after 10 minutes
          scheduleAutoCheckout(normalizedPlate);
        } catch (err) {
          // If parking session creation fails (e.g., spot already occupied), log but continue
          console.error(`[PARKING] Failed to create session: ${err.message}`);
          parkingError = err.message;
        }
      }

      // Check all registered vehicles that match this plate
      const matchingVehicles = await UserVehicle.getAllByLicensePlate(normalizedPlate);
      const locationStr = location || `Floor ${floor}, Lot ${lot}`;
      const notifications = [];
      const notificationErrors = [];

      for (const vehicle of matchingVehicles) {
        console.log(`[DETECTION] Vehicle found for plate ${normalizedPlate}, user_id: ${vehicle.user_id}`);
        const user = await User.getByIdForNotification(vehicle.user_id);
        if (!user) {
          continue;
        }

        try {
          const notification = await Notification.create(
            vehicle.user_id,
            vehicle.id,
            detection.id,
            exitEvent ? 'Vehicle Exit Detected' : 'Vehicle Detected',
            exitEvent
              ? `Your registered license plate ${normalizedPlate} has exited via ${locationStr}`
              : `Your registered license plate ${normalizedPlate} was detected at ${locationStr}`,
            locationStr,
            detection.detected_at
          );

          notifications.push(notification);
          console.log(`[DETECTION] Notification created for user ${vehicle.user_id}: ${notification.id}`);

          if (user.push_notification_enabled) {
            // TODO: Send push notification via FCM, OneSignal, or similar service
            console.log(`Push notification sent to user ${vehicle.user_id}`);
          }
        } catch (notificationError) {
          console.error(`Failed to create notification for user ${vehicle.user_id}: ${notificationError.message}`);
          notificationErrors.push({ userId: vehicle.user_id, error: notificationError.message });
        }
      }

      if (notifications.length > 0) {
        return res.status(201).json({
          success: true,
          message: exitEvent
            ? 'Exit detection recorded and user notified'
            : 'Detection recorded and user notified',
          matched: true,
          parkingSessionCreated: !!parkingSession,
          sessionClosedByExit: !!sessionClosedByExit,
          parkingError,
          data: {
            detection,
            notifications,
            parkingSession,
            sessionClosedByExit
          },
          notificationErrors
        });
      }

      res.status(201).json({
        success: true,
        message: exitEvent ? 'Exit detection recorded' : 'Detection recorded',
        parkingSessionCreated: !!parkingSession,
        sessionClosedByExit: !!sessionClosedByExit,
        parkingError,
        data: { detection, parkingSession, sessionClosedByExit },
        matched: matchingVehicles.length > 0,
        notificationErrors
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get detection history for a specific plate
  static async getPlateDetectionHistory(req, res) {
    try {
      const { licensePlate } = req.params;
      const { token } = req.query;

      if (!licensePlate) {
        return res.status(400).json({
          success: false,
          message: 'License plate is required'
        });
      }

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

      // Verify user owns this vehicle
      const vehicle = await UserVehicle.getByUserIdAndLicensePlate(user.id, licensePlate);
      if (!vehicle) {
        return res.status(403).json({
          success: false,
          message: 'Unauthorized'
        });
      }

      const detections = await DetectionEvent.getDetectionsByPlate(licensePlate);

      res.json({
        success: true,
        count: detections.length,
        data: detections
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get recent detections (admin endpoint)
  static async getRecentDetections(req, res) {
    try {
      const { minutesBack = 60, limit = 100 } = req.query;

      const detections = await DetectionEvent.getRecentDetections(
        parseInt(limit),
        parseInt(minutesBack)
      );

      res.json({
        success: true,
        count: detections.length,
        data: detections
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get today's detection count for a specific plate
  static async getDetectionCount(req, res) {
    try {
      const { licensePlate } = req.params;
      const { token } = req.query;

      if (!licensePlate) {
        return res.status(400).json({
          success: false,
          message: 'License plate is required'
        });
      }

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

      // Verify user owns this vehicle
      const vehicle = await UserVehicle.getByUserIdAndLicensePlate(user.id, licensePlate);
      if (!vehicle) {
        return res.status(403).json({
          success: false,
          message: 'Unauthorized'
        });
      }

      const count = await DetectionEvent.countDetectionsToday(licensePlate);

      res.json({
        success: true,
        data: {
          licensePlate,
          todayCount: count
        }
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // BUG FIX (helper method): Restore auto-checkout timer for a vehicle (called on server startup)
  // Re-populates the in-memory timer map when server restarts, allowing active sessions to be tracked
  static restoreTimer(licensePlate, timerId) {
    autoCheckoutTimers[licensePlate] = timerId;
  }
}

module.exports = DetectionController;
