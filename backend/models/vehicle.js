const { pool } = require('../config/db');

class UserVehicle {
  static normalizePlate(licensePlate) {
    return String(licensePlate || '')
      .toUpperCase()
      .replace(/[^A-Z0-9]/g, '')
      .trim();
  }

  static async create(userId, licensePlate, vehicleName, makeModel = null, color = null) {
    try {
      const normalizedPlate = this.normalizePlate(licensePlate);
      const result = await pool.query(
        `INSERT INTO user_vehicles (user_id, license_plate, vehicle_name, make_model, color)
         VALUES ($1, $2, $3, $4, $5)
         RETURNING id, user_id, license_plate, vehicle_name, make_model, color, created_at`,
        [userId, normalizedPlate, vehicleName, makeModel, color]
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
    const normalizedPlate = this.normalizePlate(licensePlate);
    const result = await pool.query(
      `SELECT *
       FROM user_vehicles
       WHERE UPPER(REGEXP_REPLACE(license_plate, '[^A-Z0-9]', '', 'g')) = $1
       ORDER BY created_at DESC
       LIMIT 1`,
      [normalizedPlate]
    );
    return result.rows[0] || null;
  }

  static async getAllByLicensePlate(licensePlate) {
    const normalizedPlate = this.normalizePlate(licensePlate);
    const result = await pool.query(
      `SELECT *
       FROM user_vehicles
       WHERE UPPER(REGEXP_REPLACE(license_plate, '[^A-Z0-9]', '', 'g')) = $1
       ORDER BY created_at DESC`,
      [normalizedPlate]
    );
    return result.rows;
  }

  static async getByUserIdAndLicensePlate(userId, licensePlate) {
    const normalizedPlate = this.normalizePlate(licensePlate);
    const result = await pool.query(
      `SELECT *
       FROM user_vehicles
       WHERE user_id = $1
       AND UPPER(REGEXP_REPLACE(license_plate, '[^A-Z0-9]', '', 'g')) = $2
       LIMIT 1`,
      [userId, normalizedPlate]
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

  static async getParkingHistoryByUserId(userId, limit = 100) {
    const boundedLimit = Math.max(1, Math.min(Number(limit) || 100, 500));

    const result = await pool.query(
      `SELECT
         ps.id AS session_id,
         uv.id AS vehicle_id,
         uv.vehicle_name,
         uv.license_plate,
         ps.floor,
         ps.lot,
         CONCAT('F', ps.floor, '-S', ps.lot) AS spot_number,
         CONCAT('Floor ', ps.floor) AS parking_area,
         ps.check_in_time,
         ps.check_out_time
       FROM user_vehicles uv
       INNER JOIN parking_sessions ps
         ON UPPER(REGEXP_REPLACE(uv.license_plate, '[^A-Z0-9]', '', 'g')) =
            UPPER(REGEXP_REPLACE(ps.vehicle_plate, '[^A-Z0-9]', '', 'g'))
       WHERE uv.user_id = $1
       ORDER BY ps.check_in_time DESC
       LIMIT $2`,
      [userId, boundedLimit]
    );

    return result.rows;
  }
}

module.exports = UserVehicle;
