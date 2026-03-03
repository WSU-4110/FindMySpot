const DetectionEvent = require('../models/detection');
const UserVehicle = require('../models/vehicle');
const Notification = require('../models/notification');
const { User } = require('../models/user');

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
        longitude 
      } = req.body;

      if (!licensePlate) {
        return res.status(400).json({
          success: false,
          message: 'License plate is required'
        });
      }

      // Record the detection
      const detection = await DetectionEvent.recordDetection(
        licensePlate,
        floor,
        lot,
        location,
        confidence,
        cameraId,
        latitude,
        longitude
      );

      // Check if this plate is registered
      const vehicle = await UserVehicle.getByLicensePlate(licensePlate);

      if (vehicle) {
        console.log(`[DETECTION] Vehicle found for plate ${licensePlate}, user_id: ${vehicle.user_id}`);
        // Get the user info
        const user = await User.getByIdForNotification(vehicle.user_id);
        
        if (user) {
          // Create notification for registered vehicle
          const locationStr = location || `Floor ${floor}, Lot ${lot}`;
          try {
            const notification = await Notification.create(
              vehicle.user_id,
              vehicle.id,
              detection.id,
              'Vehicle Detected',
              `Your registered license plate ${licensePlate} was detected at ${locationStr}`,
              locationStr,
              detection.detected_at
            );

            console.log(`[DETECTION] Notification created for user ${vehicle.user_id}: ${notification.id}`);

            // Only send push notification if user has enabled it
            if (user.push_notification_enabled) {
              // TODO: Send push notification via FCM, OneSignal, or similar service
              console.log(`Push notification sent to user ${vehicle.user_id}`);
            }

            return res.status(201).json({
              success: true,
              message: 'Detection recorded and user notified',
              matched: true,
              data: {
                detection,
                notification
              }
            });
          } catch (notificationError) {
            console.error(`Failed to create notification: ${notificationError.message}`);
            // Still return success for detection, but note it failed
            return res.status(201).json({
              success: true,
              message: 'Detection recorded (notification creation failed)',
              data: { detection },
              matched: true,
              notificationError: notificationError.message
            });
          }
        }
      }

      res.status(201).json({
        success: true,
        message: 'Detection recorded',
        data: { detection },
        matched: !!vehicle
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
      const vehicle = await UserVehicle.getByLicensePlate(licensePlate);
      if (!vehicle || vehicle.user_id !== user.id) {
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
      const vehicle = await UserVehicle.getByLicensePlate(licensePlate);
      if (!vehicle || vehicle.user_id !== user.id) {
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
}

module.exports = DetectionController;
