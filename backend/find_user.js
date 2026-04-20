const { Pool } = require('pg');
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'findmyspot',
  password: '1442',
  port: 5432,
});

async function findUser() {
  try {
    const res = await pool.query("SELECT u.token, uv.user_id, uv.license_plate FROM users u JOIN user_vehicles uv ON u.id = uv.user_id WHERE u.token IS NOT NULL AND u.token <> '' LIMIT 1");
    if (res.rows.length === 0) {
        console.log('null');
    } else {
        console.log(JSON.stringify(res.rows[0]));
    }
    await pool.end();
  } catch (err) {
    console.error(err);
    process.exit(1);
  }
}
findUser();
