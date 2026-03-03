const express = require('express');
const router = express.Router();
const NotificationController = require('../controllers/notificationController');

// More specific routes first
router.get('/unread/:token', NotificationController.getUnreadNotifications);
router.get('/count/:token', NotificationController.getUnreadCount);

// PUT routes (more specific first)
router.put('/:token/mark-all-read', NotificationController.markAllAsRead);
router.put('/:token/:notificationId/read', NotificationController.markAsRead);

// Generic routes last
router.get('/:token', NotificationController.getUserNotifications);
router.delete('/:token/:notificationId', NotificationController.deleteNotification);

module.exports = router;
