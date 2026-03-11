const express = require('express');
const router = express.Router();
const DetectionController = require('../controllers/detectionController');

// Detection recording (called by camera/AI system)
router.post('/record', DetectionController.recordPlateDetection);

// Get detection history for a plate
router.get('/history/:licensePlate', DetectionController.getPlateDetectionHistory);

// Get detection count today
router.get('/count/:licensePlate', DetectionController.getDetectionCount);

// Get recent detections (admin)
router.get('/recent', DetectionController.getRecentDetections);

module.exports = router;
