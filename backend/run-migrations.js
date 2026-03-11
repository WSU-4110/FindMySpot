const fs = require('fs');
const path = require('path');
require('dotenv').config();
const { pool } = require('./config/db');

async function runMigrations() {
  try {
    console.log('📦 Starting database migrations...\n');

    // Read migration files
    const migration1 = fs.readFileSync(path.join(__dirname, '../database/migrations/001_init.sql'), 'utf8');
    const migration2 = fs.readFileSync(path.join(__dirname, '../database/migrations/002_license_plate_detection.sql'), 'utf8');
    const migration3 = fs.readFileSync(path.join(__dirname, '../database/migrations/003_security_flags.sql'), 'utf8');

    // Run migration 1
    console.log('🔧 Running migration 001_init.sql...');
    await pool.query(migration1);
    console.log('✓ Migration 001_init.sql completed\n');

    // Run migration 2
    console.log('🔧 Running migration 002_license_plate_detection.sql...');
    await pool.query(migration2);
    console.log('✓ Migration 002_license_plate_detection.sql completed\n');

    // Run migration 3
    console.log('🔧 Running migration 003_security_flags.sql...');
    await pool.query(migration3);
    console.log('✓ Migration 003_security_flags.sql completed\n');

    console.log('✅ All migrations completed successfully!');
    console.log('\n📊 Verifying tables...');

    // Verify tables exist
    const result = await pool.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      ORDER BY table_name;
    `);

    console.log('\nCreated tables:');
    result.rows.forEach(row => {
      console.log(`  • ${row.table_name}`);
    });

    await pool.end();
    process.exit(0);
  } catch (error) {
    console.error('❌ Migration failed:', error.message);
    await pool.end();
    process.exit(1);
  }
}

runMigrations();
