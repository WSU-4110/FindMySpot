const UserVehicle = require('../models/vehicle');
const { User } = require('../models/user');

class VehicleController {
  // Register a new vehicle for the user
  static async registerVehicle(req, res) {
    try {
      const { token, licensePlate, vehicleName, makeModel, color } = req.body;

      if (!token || !licensePlate) {
        return res.status(400).json({
          success: false,
          message: 'Token and license plate are required'
        });
      }

      const user = await User.getByToken(token);
      console.log(`[VEHICLE] Registering vehicle for token ${token.substring(0, 20)}... - User: ${user ? user.id : 'NOT FOUND'}`);
      
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      console.log(`[VEHICLE] Creating vehicle ${licensePlate} for user ${user.id}`);
      
      const vehicle = await UserVehicle.create(
        user.id,
        licensePlate,
        vehicleName || licensePlate,
        makeModel,
        color
      );

      console.log(`[VEHICLE] Created vehicle ${vehicle.id} with user_id ${vehicle.user_id}`);

      res.status(201).json({
        success: true,
        message: 'Vehicle registered successfully',
        data: vehicle
      });
    } catch (error) {
      if (error.message.includes('already registered')) {
        return res.status(400).json({
          success: false,
          message: error.message
        });
      }
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get all vehicles for the user
  static async getUserVehicles(req, res) {
    try {
      const { token } = req.params;

      if (!token) {
        return res.status(400).json({
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

      const vehicles = await UserVehicle.getByUserId(user.id);

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

  // Get parking session history for the user's saved vehicles
  static async getUserParkingHistory(req, res) {
    try {
      const { token } = req.params;
      const { limit = 100 } = req.query;

      if (!token) {
        return res.status(400).json({
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

      const history = await UserVehicle.getParkingHistoryByUserId(user.id, parseInt(limit, 10));

      res.json({
        success: true,
        count: history.length,
        data: history
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Update a vehicle
  static async updateVehicle(req, res) {
    try {
      const { token, vehicleId } = req.params;
      const { vehicleName, makeModel, color } = req.body;

      if (!token || !vehicleId) {
        return res.status(400).json({
          success: false,
          message: 'Token and vehicle ID are required'
        });
      }

      const user = await User.getByToken(token);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      const vehicle = await UserVehicle.update(
        vehicleId,
        user.id,
        vehicleName,
        makeModel,
        color
      );

      if (!vehicle) {
        return res.status(404).json({
          success: false,
          message: 'Vehicle not found'
        });
      }

      res.json({
        success: true,
        message: 'Vehicle updated successfully',
        data: vehicle
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Delete a vehicle
  static async deleteVehicle(req, res) {
    try {
      const { vehicleId } = req.params;
      const { token } = req.body;

      if (!token || !vehicleId) {
        return res.status(400).json({
          success: false,
          message: 'Token and vehicle ID are required'
        });
      }

      const user = await User.getByToken(token);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      const result = await UserVehicle.delete(vehicleId, user.id);

      if (!result) {
        return res.status(404).json({
          success: false,
          message: 'Vehicle not found'
        });
      }

      res.json({
        success: true,
        message: 'Vehicle deleted successfully'
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }
}

module.exports = VehicleController;
