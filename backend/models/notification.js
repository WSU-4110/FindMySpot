const { pool } = require('../config/db');

class Notification {
  static async create(userId, vehicleId, detectedPlateId, title, message, locationDescription = null, detectedAt = null) {
    try {
      const result = await pool.query(
        `INSERT INTO notifications (user_id, vehicle_id, detected_plate_id, title, message, location_description, detected_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         RETURNING *`,
        [userId, vehicleId, detectedPlateId, title, message, locationDescription, detectedAt]
      );
      return result.rows[0];
    } catch (error) {
      throw error;
    }
  }

  static async getByUserId(userId, limit = 50, offset = 0) {
    const result = await pool.query(
      `SELECT n.*, uv.license_plate, uv.vehicle_name
       FROM notifications n
       LEFT JOIN user_vehicles uv ON n.vehicle_id = uv.id
       WHERE n.user_id = $1
       ORDER BY n.sent_at DESC
       LIMIT $2 OFFSET $3`,
      [userId, limit, offset]
    );
    return result.rows;
  }

  static async getUnreadByUserId(userId) {
    const result = await pool.query(
      `SELECT n.*, uv.license_plate, uv.vehicle_name
       FROM notifications n
       LEFT JOIN user_vehicles uv ON n.vehicle_id = uv.id
       WHERE n.user_id = $1 AND n.is_read = false
       ORDER BY n.sent_at DESC`,
      [userId]
    );
    return result.rows;
  }

  static async markAsRead(notificationId, userId) {
    const result = await pool.query(
      `UPDATE notifications 
       SET is_read = true, read_at = CURRENT_TIMESTAMP
       WHERE id = $1 AND user_id = $2
       RETURNING *`,
      [notificationId, userId]
    );
    return result.rows[0] || null;
  }

  static async markAllAsRead(userId) {
    const result = await pool.query(
      `UPDATE notifications 
       SET is_read = true, read_at = CURRENT_TIMESTAMP
       WHERE user_id = $1 AND is_read = false
       RETURNING id`,
      [userId]
    );
    return result.rows.length;
  }

  static async getUnreadCount(userId) {
    const result = await pool.query(
      `SELECT COUNT(*) as count FROM notifications 
       WHERE user_id = $1 AND is_read = false`,
      [userId]
    );
    return parseInt(result.rows[0].count);
  }

  static async delete(notificationId, userId) {
    const result = await pool.query(
      `DELETE FROM notifications WHERE id = $1 AND user_id = $2 RETURNING id`,
      [notificationId, userId]
    );
    return result.rows[0] || null;
  }
}

module.exports = Notification;
