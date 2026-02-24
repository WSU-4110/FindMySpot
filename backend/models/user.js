const crypto = require('crypto');

const users = [];

class User {
  static create(name, email, password) {
    // Check if email already exists
    if (users.find(u => u.email === email)) {
      return null;
    }

    // Hash password (simple hash for demo - use bcrypt in production)
    const hashedPassword = crypto.createHash('sha256').update(password).digest('hex');
    const token = crypto.randomBytes(32).toString('hex');

    const user = {
      id: users.length + 1,
      name,
      email,
      password: hashedPassword,
      token,
      createdAt: new Date().toISOString()
    };

    users.push(user);
    return user;
  }

  static authenticate(email, password) {
    const hashedPassword = crypto.createHash('sha256').update(password).digest('hex');
    const user = users.find(u => u.email === email && u.password === hashedPassword);
    
    if (user) {
      // Generate new token on login
      user.token = crypto.randomBytes(32).toString('hex');
    }
    
    return user;
  }

  static getByToken(token) {
    return users.find(u => u.token === token);
  }

  static getByEmail(email) {
    return users.find(u => u.email === email);
  }

  static getById(id) {
    return users.find(u => u.id === id);
  }

  static getAll() {
    return users.map(u => ({
      id: u.id,
      name: u.name,
      email: u.email,
      createdAt: u.createdAt
    }));
  }
}

module.exports = { User };
