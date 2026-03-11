const { pool } = require('../config/db');

class DetectionEvent {
  static async recordDetection(licensePlate, floor = null, lot = null, locationDescription = null, 
                               confidence = 0.95, cameraId = null, latitude = null, longitude = null) {
    try {
      const result = await pool.query(
        `INSERT INTO detected_plates (license_plate, floor, lot, location_description, confidence, camera_id, latitude, longitude)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         RETURNING *`,
        [licensePlate.toUpperCase(), floor, lot, locationDescription, confidence, cameraId, latitude, longitude]
      );
      return result.rows[0];
    } catch (error) {
      throw error;
    }
  }

  static async getRecentDetections(limit = 100, minutesBack = 60) {
    const result = await pool.query(
      `SELECT * FROM detected_plates 
       WHERE detected_at > NOW() - INTERVAL '${minutesBack} minutes'
       ORDER BY detected_at DESC
       LIMIT $1`,
      [limit]
    );
    return result.rows;
  }

  static async getDetectionsByPlate(licensePlate, limit = 50) {
    const result = await pool.query(
      `SELECT * FROM detected_plates 
       WHERE license_plate = $1
       ORDER BY detected_at DESC
       LIMIT $2`,
      [licensePlate.toUpperCase(), limit]
    );
    return result.rows;
  }

  static async getDetectionById(detectionId) {
    const result = await pool.query(
      `SELECT * FROM detected_plates WHERE id = $1`,
      [detectionId]
    );
    return result.rows[0] || null;
  }

  static async getDetectionsSince(timestamp, limit = 100) {
    const result = await pool.query(
      `SELECT * FROM detected_plates 
       WHERE detected_at > $1
       ORDER BY detected_at DESC
       LIMIT $2`,
      [timestamp, limit]
    );
    return result.rows;
  }

  static async countDetectionsToday(licensePlate) {
    const result = await pool.query(
      `SELECT COUNT(*) as count FROM detected_plates 
       WHERE license_plate = $1 
       AND DATE(detected_at) = CURRENT_DATE`,
      [licensePlate.toUpperCase()]
    );
    return parseInt(result.rows[0].count);
  }
}

module.exports = DetectionEvent;
