const crypto = require('crypto');
const { pool } = require('../config/db');

class User {
  static async create(name, email, password) {
    const hashedPassword = crypto.createHash('sha256').update(password).digest('hex');
    const token = crypto.randomBytes(32).toString('hex');

    try {
      const result = await pool.query(
        `INSERT INTO users (name, email, password, token)
         VALUES ($1, $2, $3, $4)
         RETURNING id, name, email, token`,
        [name, email, hashedPassword, token]
      );

      return result.rows[0];
    } catch (error) {
      if (error.code === '23505') {
        return null;
      }
      throw error;
    }
  }

  static async authenticate(email, password) {
    const hashedPassword = crypto.createHash('sha256').update(password).digest('hex');
    const newToken = crypto.randomBytes(32).toString('hex');

    const result = await pool.query(
      `UPDATE users
       SET token = $1
       WHERE email = $2 AND password = $3
       RETURNING id, name, email, token`,
      [newToken, email, hashedPassword]
    );

    return result.rows[0] || null;
  }

  static async getByToken(token) {
    const result = await pool.query(
      'SELECT id, name, email FROM users WHERE token = $1',
      [token]
    );

    return result.rows[0] || null;
  }

  static async getByEmail(email) {
    const result = await pool.query(
      'SELECT id, name, email FROM users WHERE email = $1',
      [email]
    );

    return result.rows[0] || null;
  }

  static async getById(id) {
    const result = await pool.query(
      'SELECT id, name, email FROM users WHERE id = $1',
      [id]
    );

    return result.rows[0] || null;
  }

  static async getAll() {
    const result = await pool.query(
      'SELECT id, name, email, created_at FROM users ORDER BY id ASC'
    );

    return result.rows;
  }
}

module.exports = { User };
