const express = require('express');
const router = express.Router();
const VehicleController = require('../controllers/vehicleController');

// Vehicle management routes
router.post('/', VehicleController.registerVehicle);  // POST with token in body
router.get('/:token/history', VehicleController.getUserParkingHistory);
router.get('/:token', VehicleController.getUserVehicles);
router.put('/:token/:vehicleId', VehicleController.updateVehicle);
router.delete('/:vehicleId', VehicleController.deleteVehicle);  // DELETE with token in body

module.exports = router;
