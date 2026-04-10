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
      byFloor,
      generatedAt: new Date().toISOString()
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
    const normalizedPlate = String(vehiclePlate || '').toUpperCase().trim();

    const activeSession = await this.getActiveByVehicle(normalizedPlate);
    if (activeSession) {
      throw new Error(`Vehicle already has an active session: ${normalizedPlate}`);
    }

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
      [normalizedPlate, floor, lot]
    );

    const session = result.rows[0];

    // Update spot occupancy
    await ParkingSpot.updateOccupancy(floor, lot, true, vehiclePlate);

    return session;
  }

  static async checkout(vehiclePlate) {
    const normalizedPlate = String(vehiclePlate || '').toUpperCase().trim();

    // Find active session
    const result = await pool.query(
      `UPDATE parking_sessions
       SET check_out_time = CURRENT_TIMESTAMP
       WHERE vehicle_plate = $1 AND check_out_time IS NULL
       RETURNING id, vehicle_plate, floor, lot, check_in_time, check_out_time`,
      [normalizedPlate]
    );

    if (result.rows.length === 0) {
      throw new Error(`No active session found for vehicle: ${normalizedPlate}`);
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
      [String(vehiclePlate || '').toUpperCase().trim()]
    );
    return result.rows;
  }

  static async getActiveByVehicle(vehiclePlate) {
    const result = await pool.query(
      `SELECT id, vehicle_plate, floor, lot, check_in_time, check_out_time
       FROM parking_sessions
       WHERE vehicle_plate = $1 AND check_out_time IS NULL
       ORDER BY check_in_time DESC
       LIMIT 1`,
      [String(vehiclePlate || '').toUpperCase().trim()]
    );
    return result.rows[0] || null;
  }

  static async locateVehicle(vehiclePlate) {
    const normalizedPlate = String(vehiclePlate || '').toUpperCase().trim();

    const activeSessionResult = await pool.query(
      `SELECT id, vehicle_plate, floor, lot, check_in_time, check_out_time
       FROM parking_sessions
       WHERE vehicle_plate = $1 AND check_out_time IS NULL
       ORDER BY check_in_time DESC
       LIMIT 1`,
      [normalizedPlate]
    );

    const session = activeSessionResult.rows[0] || null;

    if (!session) {
      return null;
    }

    const detectionResult = await pool.query(
      `SELECT id, floor, lot, location_description, detected_at
       FROM detected_plates
       WHERE license_plate = $1
       ORDER BY detected_at DESC
       LIMIT 1`,
      [normalizedPlate]
    );

    const latestDetection = detectionResult.rows[0] || null;

    const floor = session.floor ?? latestDetection?.floor ?? null;
    const lot = session.lot ?? latestDetection?.lot ?? null;
    const spotNumber = floor != null && lot != null ? `F${floor}-S${lot}` : null;
    const area = floor != null ? `Floor ${floor}` : null;
    const locationDescription = latestDetection?.location_description || null;

    return {
      vehiclePlate: normalizedPlate,
      sessionId: session.id,
      floor,
      lot,
      spotNumber,
      area,
      locationDescription,
      checkInTime: session.check_in_time || null,
      checkOutTime: session.check_out_time || null,
      sessionActive: true,
      source: 'parking_session',
      spotDataMissing: floor == null || lot == null,
      locationOnly: floor == null && lot == null,
      detectedAt: latestDetection?.detected_at || null
    };
  }

  static async getDirectionsForVehicle(vehiclePlate) {
    const location = await this.locateVehicle(vehiclePlate);
    if (!location) {
      return null;
    }

    const hasPreciseSpot = location.floor != null && location.lot != null;
    if (!hasPreciseSpot) {
      return {
        navigationAvailable: false,
        fallback: true,
        location,
        map: null,
        steps: []
      };
    }

    const zone = String.fromCharCode(65 + ((Number(location.lot) - 1) % 3));
    const steps = [
      `Go to ${location.area}.`,
      `Proceed to Zone ${zone}.`,
      `Find Spot ${location.spotNumber}.`
    ];

    return {
      navigationAvailable: true,
      fallback: false,
      location,
      map: {
        floor: location.floor,
        zone,
        lot: location.lot,
        spotNumber: location.spotNumber
      },
      steps
    };
  }

  static async getUsageReport(hoursBack = 24) {
    const boundedHours = Math.max(1, Math.min(Number(hoursBack) || 24, 24 * 30));

    const summaryResult = await pool.query(
      `SELECT
        COUNT(*)::int AS total_sessions,
        COUNT(*) FILTER (WHERE check_out_time IS NOT NULL)::int AS completed_sessions,
        ROUND(AVG(EXTRACT(EPOCH FROM (check_out_time - check_in_time)) / 60)
          FILTER (WHERE check_out_time IS NOT NULL), 2) AS avg_duration_minutes
       FROM parking_sessions
       WHERE check_in_time >= NOW() - ($1::text || ' hours')::interval`,
      [String(boundedHours)]
    );

    const peakUsageResult = await pool.query(
      `SELECT
        EXTRACT(HOUR FROM check_in_time)::int AS hour,
        COUNT(*)::int AS session_count
       FROM parking_sessions
       WHERE check_in_time >= NOW() - ($1::text || ' hours')::interval
       GROUP BY EXTRACT(HOUR FROM check_in_time)
       ORDER BY session_count DESC, hour ASC
       LIMIT 5`,
      [String(boundedHours)]
    );

    const occupancySnapshot = await ParkingSpot.getOccupancyStats();
    const summary = summaryResult.rows[0] || {};

    return {
      windowHours: boundedHours,
      generatedAt: new Date().toISOString(),
      metrics: {
        totalSessions: Number(summary.total_sessions || 0),
        completedSessions: Number(summary.completed_sessions || 0),
        averageParkingDurationMinutes: summary.avg_duration_minutes == null
          ? null
          : Number(summary.avg_duration_minutes),
        occupancyRate: occupancySnapshot.total === 0
          ? 0
          : Number(((occupancySnapshot.occupied / occupancySnapshot.total) * 100).toFixed(2))
      },
      peakUsageHours: peakUsageResult.rows,
      occupancySnapshot
    };
  }

  static async scanAndFlagSecurityIssues(maxDurationHours = 24) {
    const boundedHours = Math.max(1, Math.min(Number(maxDurationHours) || 24, 24 * 14));

    const activeSessionsResult = await pool.query(
      `SELECT
        ps.id,
        ps.vehicle_plate,
        ps.floor,
        ps.lot,
        ps.check_in_time,
        EXTRACT(EPOCH FROM (NOW() - ps.check_in_time)) / 3600 AS parked_hours,
        (SELECT COUNT(*)::int FROM user_vehicles uv
          WHERE UPPER(REGEXP_REPLACE(uv.license_plate, '[^A-Z0-9]', '', 'g')) =
                UPPER(REGEXP_REPLACE(ps.vehicle_plate, '[^A-Z0-9]', '', 'g'))
        ) AS authorization_records
       FROM parking_sessions ps
       WHERE ps.check_out_time IS NULL
       ORDER BY ps.check_in_time ASC`
    );

    const createdFlags = [];

    for (const session of activeSessionsResult.rows) {
      if (Number(session.parked_hours) >= boundedHours) {
        const longTermFlag = await SecurityFlag.createIfMissing(
          session.id,
          session.vehicle_plate,
          'LONG_TERM_STATIONARY',
          `Vehicle has been stationary for ${Number(session.parked_hours).toFixed(2)} hours`
        );
        if (longTermFlag) {
          createdFlags.push(longTermFlag);
        }
      }

      if (Number(session.authorization_records) === 0) {
        const authFlag = await SecurityFlag.createIfMissing(
          session.id,
          session.vehicle_plate,
          'AUTHORIZATION_DATA_MISSING',
          'No authorization data found for this active session; flagged for review.'
        );
        if (authFlag) {
          createdFlags.push(authFlag);
        }
      }
    }

    return {
      scannedSessions: activeSessionsResult.rows.length,
      flagsCreated: createdFlags.length,
      createdFlags
    };
  }

  static async closeByExitDetection(vehiclePlate, detectionMeta = {}) {
    const session = await this.checkout(vehiclePlate);
    return {
      ...session,
      closedBy: 'exit_detection',
      detectionMeta
    };
  }

  static async getAll() {
    const result = await pool.query(
      'SELECT id, vehicle_plate, floor, lot, check_in_time, check_out_time FROM parking_sessions ORDER BY check_in_time DESC'
    );
    return result.rows;
  }
}

class SecurityFlag {
  static async createIfMissing(sessionId, vehiclePlate, flagType, reason) {
    const existingResult = await pool.query(
      `SELECT id, session_id, vehicle_plate, flag_type, reason, status, created_at, resolved_at
       FROM security_flags
       WHERE session_id = $1
         AND vehicle_plate = $2
         AND flag_type = $3
         AND status = 'OPEN'
       LIMIT 1`,
      [sessionId, String(vehiclePlate || '').toUpperCase().trim(), flagType]
    );

    if (existingResult.rows[0]) {
      return null;
    }

    const insertResult = await pool.query(
      `INSERT INTO security_flags (session_id, vehicle_plate, flag_type, reason, status)
       VALUES ($1, $2, $3, $4, 'OPEN')
       RETURNING id, session_id, vehicle_plate, flag_type, reason, status, created_at, resolved_at`,
      [sessionId, String(vehiclePlate || '').toUpperCase().trim(), flagType, reason]
    );

    return insertResult.rows[0] || null;
  }

  static async getOpen(limit = 100) {
    const safeLimit = Math.max(1, Math.min(Number(limit) || 100, 500));
    const result = await pool.query(
      `SELECT id, session_id, vehicle_plate, flag_type, reason, status, created_at, resolved_at
       FROM security_flags
       WHERE status = 'OPEN'
       ORDER BY created_at DESC
       LIMIT $1`,
      [safeLimit]
    );
    return result.rows;
  }
}

module.exports = {
  ParkingSpot,
  Vehicle,
  ParkingSession,
  SecurityFlag
};