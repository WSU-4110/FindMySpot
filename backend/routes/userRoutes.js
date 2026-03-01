const express = require('express');
const router = express.Router();
const UserController = require('../controllers/userController');

// Authentication routes
router.post('/register', UserController.register);
router.post('/login', UserController.login);
router.post('/verify-token', UserController.verifyToken);
router.get('/profile/:token', UserController.getProfile);

module.exports = router;
