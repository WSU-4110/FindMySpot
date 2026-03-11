const express = require('express');
const router = express.Router();
const ParkingController = require('../controllers/parkingController');

// Parking spot routes
router.get('/spots', ParkingController.getAllSpots);
router.get('/spots/available', ParkingController.getAvailableSpots);
router.get('/spots/occupied', ParkingController.getOccupiedSpots);
router.get('/spots/floor/:floor', ParkingController.getSpotsByFloor);
router.get('/spots/floor/:floor/lot/:lot', ParkingController.getSpot);
router.get('/stats', ParkingController.getOccupancyStats);
router.get('/stats/live', ParkingController.getOccupancyStats);
router.get('/reports/usage', ParkingController.getUsageAnalyticsReport);
router.get('/security/flags', ParkingController.getOpenSecurityFlags);
router.post('/security/scan', ParkingController.runSecurityFlagScan);

// Parking session routes
router.post('/checkin', ParkingController.checkIn);
router.post('/checkout', ParkingController.checkOut);
router.get('/sessions/active', ParkingController.getActiveSessions);
router.get('/locate/:plate', ParkingController.locateVehicleWithSpot);
router.get('/directions/:plate', ParkingController.getVehicleDirections);

// Vehicle routes
router.get('/vehicles', ParkingController.getAllVehicles);
router.get('/vehicles/:plate', ParkingController.getVehicle);

module.exports = router;
