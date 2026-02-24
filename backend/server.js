const express = require('express');
const cors = require('cors');
const parkingRoutes = require('./routes/parkingRoutes');
const userRoutes = require('./routes/userRoutes');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/parking', parkingRoutes);
app.use('/api/auth', userRoutes);

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
      spots: '/api/parking/spots',
      available: '/api/parking/spots/available',
      occupied: '/api/parking/spots/occupied',
      stats: '/api/parking/stats',
      checkin: 'POST /api/parking/checkin',
      checkout: 'POST /api/parking/checkout',
      sessions: '/api/parking/sessions/active',
      vehicles: '/api/parking/vehicles'
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
app.listen(PORT, () => {
  console.log(`FindMySpot API server running on port ${PORT}`);
  console.log(`Visit http://localhost:${PORT} for API documentation`);
});

module.exports = app;
