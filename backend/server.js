const express = require('express');
const cors = require('cors');
require('dotenv').config();
const parkingRoutes = require('./routes/parkingRoutes');
const userRoutes = require('./routes/userRoutes');
const vehicleRoutes = require('./routes/vehicleRoutes');
const detectionRoutes = require('./routes/detectionRoutes');
const notificationRoutes = require('./routes/notificationRoutes');
const { initializeDatabase } = require('./config/db');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/parking', parkingRoutes);
app.use('/api/auth', userRoutes);
app.use('/api/vehicles', vehicleRoutes);
app.use('/api/detection', detectionRoutes);
app.use('/api/notifications', notificationRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', message: 'FindMySpot API is running' });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'FindMySpot Parking Management API',
    version: '1.0.0',
    endpoints: {
      auth: {
        register: 'POST /api/auth/register',
        login: 'POST /api/auth/login',
        profile: 'GET /api/auth/profile/:token'
      },
      vehicles: {
        register: 'POST /api/vehicles/register/:token',
        getAll: 'GET /api/vehicles/:token',
        update: 'PUT /api/vehicles/:token/:vehicleId',
        delete: 'DELETE /api/vehicles/:token/:vehicleId'
      },
      detection: {
        record: 'POST /api/detection/record',
        history: 'GET /api/detection/history/:licensePlate',
        count: 'GET /api/detection/count/:licensePlate',
        recent: 'GET /api/detection/recent'
      },
      notifications: {
        getAll: 'GET /api/notifications/:token',
        getUnread: 'GET /api/notifications/unread/:token',
        getCount: 'GET /api/notifications/count/:token',
        markRead: 'PUT /api/notifications/:token/:notificationId/read',
        markAllRead: 'PUT /api/notifications/:token/mark-all-read',
        delete: 'DELETE /api/notifications/:token/:notificationId'
      },
      parking: {
        spots: 'GET /api/parking/spots',
        available: 'GET /api/parking/spots/available',
        occupied: 'GET /api/parking/spots/occupied',
        stats: 'GET /api/parking/stats'
      }
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    message: 'Something went wrong!',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found'
  });
});

// Start server
async function startServer() {
  try {
    await initializeDatabase();
    app.listen(PORT, () => {
      console.log(`FindMySpot API server running on port ${PORT}`);
      console.log(`Visit http://localhost:${PORT} for API documentation`);
    });
  } catch (error) {
    console.error('Failed to initialize database:', error.message);
    process.exit(1);
  }
}

startServer();

module.exports = app;
