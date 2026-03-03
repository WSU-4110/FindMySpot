const { pool } = require('../config/db');

class ParkingSpot {
  static async getAll() {
    const result = await pool.query('SELECT * FROM parking_spots ORDER BY floor, lot');
    return result.rows;
  }

  static async getByFloorAndLot(floor, lot) {
    const result = await pool.query(
      'SELECT * FROM parking_spots WHERE floor = $1 AND lot = $2',
      [floor, lot]
    );
    return result.rows[0] || null;
  }

  static async getByFloor(floor) {
    const result = await pool.query(
      'SELECT * FROM parking_spots WHERE floor = $1 ORDER BY lot',
      [floor]
    );
    return result.rows;
  }

  static async getAvailable() {
    const result = await pool.query(
      'SELECT * FROM parking_spots WHERE occupied = false ORDER BY floor, lot'
    );
    return result.rows;
  }

  static async getOccupied() {
    const result = await pool.query(
      'SELECT * FROM parking_spots WHERE occupied = true ORDER BY floor, lot'
    );
    return result.rows;
  }

  static async updateOccupancy(floor, lot, occupied, vehiclePlate = null) {
    const result = await pool.query(
      `UPDATE parking_spots 
       SET occupied = $1, vehicle_plate = $2, check_in_time = $3
       WHERE floor = $4 AND lot = $5
       RETURNING *`,
      [occupied, vehiclePlate, occupied ? new Date() : null, floor, lot]
    );
    return result.rows[0] || null;
  }

  static async getOccupancyStats() {
    const result = await pool.query(`
      SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN occupied THEN 1 END) as occupied_count,
        floor
      FROM parking_spots
      GROUP BY floor
      ORDER BY floor
    `);

    const byFloor = {};
    let totalSpots = 0;
    let totalOccupied = 0;

    result.rows.forEach(row => {
      const total = parseInt(row.total);
      const occupied = parseInt(row.occupied_count || 0);
      byFloor[row.floor] = {
        total,
        occupied,
        available: total - occupied
      };
      totalSpots += total;
      totalOccupied += occupied;
    });

    return {
      total: totalSpots,
      occupied: totalOccupied,
      available: totalSpots - totalOccupied,
      byFloor
    };
  }
}

class Vehicle {
  static async create(plate) {
    try {
      const result = await pool.query(
        'INSERT INTO vehicles (plate) VALUES ($1) ON CONFLICT (plate) DO UPDATE SET plate = $1 RETURNING id, plate, created_at',
        [plate]
      );
      return result.rows[0];
    } catch (error) {
      const existing = await this.getByPlate(plate);
      return existing;
    }
  }

  static async getByPlate(plate) {
    const result = await pool.query(
      'SELECT id, plate, created_at FROM vehicles WHERE plate = $1',
      [plate]
    );
    return result.rows[0] || null;
  }

  static async getAll() {
    const result = await pool.query('SELECT id, plate, created_at FROM vehicles ORDER BY id');
    return result.rows;
  }
}

class ParkingSession {
  static async create(vehiclePlate, floor, lot) {
    // Verify spot exists and is available
    const spot = await ParkingSpot.getByFloorAndLot(floor, lot);
    if (!spot) {
      throw new Error(`Spot not found: Floor ${floor}, Lot ${lot}`);
    }

    if (spot.occupied) {
      throw new Error(`Spot already occupied: Floor ${floor}, Lot ${lot}`);
    }

    // Create vehicle record
    await Vehicle.create(vehiclePlate);

    // Create parking session
    const result = await pool.query(
      `INSERT INTO parking_sessions (vehicle_plate, floor, lot, check_in_time)
       VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
       RETURNING id, vehicle_plate, floor, lot, check_in_time, check_out_time`,
      [vehiclePlate, floor, lot]
    );

    const session = result.rows[0];

    // Update spot occupancy
    await ParkingSpot.updateOccupancy(floor, lot, true, vehiclePlate);

    return session;
  }

  static async checkout(vehiclePlate) {
    // Find active session
    const result = await pool.query(
      `UPDATE parking_sessions
       SET check_out_time = CURRENT_TIMESTAMP
       WHERE vehicle_plate = $1 AND check_out_time IS NULL
       RETURNING id, vehicle_plate, floor, lot, check_in_time, check_out_time`,
      [vehiclePlate]
    );

    if (result.rows.length === 0) {
      throw new Error(`No active session found for vehicle: ${vehiclePlate}`);
    }

    const session = result.rows[0];
    // Update spot occupancy
    await ParkingSpot.updateOccupancy(session.floor, session.lot, false, null);

    return session;
  }

  static async getActive() {
    const result = await pool.query(
      `SELECT id, vehicle_plate, floor, lot, check_in_time, check_out_time
       FROM parking_sessions
       WHERE check_out_time IS NULL
       ORDER BY check_in_time DESC`
    );
    return result.rows;
  }

  static async getByVehicle(vehiclePlate) {
    const result = await pool.query(
      `SELECT id, vehicle_plate, floor, lot, check_in_time, check_out_time
       FROM parking_sessions
       WHERE vehicle_plate = $1
       ORDER BY check_in_time DESC`,
      [vehiclePlate]
    );
    return result.rows;
  }

  static async getAll() {
    const result = await pool.query(
      'SELECT id, vehicle_plate, floor, lot, check_in_time, check_out_time FROM parking_sessions ORDER BY check_in_time DESC'
    );
    return result.rows;
  }
}

module.exports = {
  ParkingSpot,
  Vehicle,
  ParkingSession
};
