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
}

module.exports = {
  pool,
  initializeDatabase
};