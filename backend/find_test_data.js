require("dotenv").config();
const { Pool } = require("pg");

const pool = new Pool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME
});

async function findUserAndPlate() {
  try {
    const res = await pool.query(`
      SELECT u.token, uv.license_plate 
      FROM user_vehicles uv 
      JOIN users u ON uv.user_id = u.id 
      WHERE u.token IS NOT NULL 
      LIMIT 1
    `);
    if (res.rows.length > 0) {
      console.log(JSON.stringify(res.rows[0]));
    } else {
      console.log("No user found with token and plate.");
    }
  } catch (err) {
    console.error(err);
  } finally {
    await pool.end();
  }
}

findUserAndPlate();
