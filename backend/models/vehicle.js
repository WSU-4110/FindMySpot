const { pool } = require('../config/db');

class UserVehicle {
  static async create(userId, licensePlate, vehicleName, makeModel = null, color = null) {
    try {
      const result = await pool.query(
        `INSERT INTO user_vehicles (user_id, license_plate, vehicle_name, make_model, color)
         VALUES ($1, $2, $3, $4, $5)
         RETURNING id, user_id, license_plate, vehicle_name, make_model, color, created_at`,
        [userId, licensePlate.toUpperCase(), vehicleName, makeModel, color]
      );
      return result.rows[0];
    } catch (error) {
      if (error.code === '23505') {
        throw new Error('This license plate is already registered');
      }
      throw error;
    }
  }

  static async getByUserId(userId) {
    const result = await pool.query(
      `SELECT * FROM user_vehicles WHERE user_id = $1 ORDER BY created_at DESC`,
      [userId]
    );
    return result.rows;
  }

  static async getByIdAndUserId(vehicleId, userId) {
    const result = await pool.query(
      `SELECT * FROM user_vehicles WHERE id = $1 AND user_id = $2`,
      [vehicleId, userId]
    );
    return result.rows[0] || null;
  }

  static async getByLicensePlate(licensePlate) {
    const result = await pool.query(
      `SELECT * FROM user_vehicles WHERE license_plate = $1`,
      [licensePlate.toUpperCase()]
    );
    return result.rows[0] || null;
  }

  static async delete(vehicleId, userId) {
    const result = await pool.query(
      `DELETE FROM user_vehicles WHERE id = $1 AND user_id = $2 RETURNING id`,
      [vehicleId, userId]
    );
    return result.rows[0] || null;
  }

  static async update(vehicleId, userId, vehicleName, makeModel, color) {
    const result = await pool.query(
      `UPDATE user_vehicles 
       SET vehicle_name = COALESCE($2, vehicle_name),
           make_model = COALESCE($3, make_model),
           color = COALESCE($4, color),
           updated_at = CURRENT_TIMESTAMP
       WHERE id = $1 AND user_id = $5
       RETURNING *`,
      [vehicleId, vehicleName, makeModel, color, userId]
    );
    return result.rows[0] || null;
  }
}

module.exports = UserVehicle;
