const { Pool } = require('pg');

if (process.env.DB_PASSWORD == null) {
  throw new Error('Missing DB_PASSWORD. Create backend/.env from backend/.env.example and set PostgreSQL credentials.');
}

const pool = new Pool({
  host: String(process.env.DB_HOST || 'localhost'),
  port: Number(process.env.DB_PORT || 5432),
  user: String(process.env.DB_USER || 'postgres'),
  password: String(process.env.DB_PASSWORD ?? ''),
  database: String(process.env.DB_NAME || 'findmyspot')
});

async function initializeDatabase() {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(64) NOT NULL,
        token VARCHAR(64),
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS vehicles (
        id SERIAL PRIMARY KEY,
        plate VARCHAR(16) UNIQUE NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS parking_spots (
        id SERIAL PRIMARY KEY,
        floor INTEGER NOT NULL,
        lot INTEGER NOT NULL,
        occupied BOOLEAN DEFAULT false,
        vehicle_plate VARCHAR(16),
        check_in_time TIMESTAMP,
        CONSTRAINT uniq_floor_lot UNIQUE (floor, lot)
      )
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS parking_sessions (
        id SERIAL PRIMARY KEY,
        vehicle_plate VARCHAR(16) NOT NULL,
        floor INTEGER NOT NULL,
        lot INTEGER NOT NULL,
        check_in_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        check_out_time TIMESTAMP
      )
    `);

    // Clear and reinitialize parking spots
    await pool.query('TRUNCATE TABLE parking_spots, parking_sessions CASCADE');
    
    await pool.query(`
      INSERT INTO parking_spots (floor, lot, occupied) 
      SELECT f.floor, l.lot, false
      FROM generate_series(1, 5) f(floor)
      CROSS JOIN generate_series(1, 5) l(lot)
    `);

    console.log('Database tables initialized successfully');
  } catch (error) {
    console.error('Database initialization error:', error.message);
    throw error;
  }
}

module.exports = {
  pool,
  initializeDatabase
};