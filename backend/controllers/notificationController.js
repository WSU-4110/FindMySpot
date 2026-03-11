const Notification = require('../models/notification');
const { User } = require('../models/user');

class NotificationController {
  // Get all notifications for the user
  static async getUserNotifications(req, res) {
    try {
      const { token } = req.params;
      const { limit = 50, offset = 0 } = req.query;

      if (!token) {
        return res.status(400).json({
          success: false,
          message: 'Token is required'
        });
      }

      const user = await User.getByToken(token);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      console.log(`[NOTIFICATION] Getting notifications for user ${user.id}`);

      const notifications = await Notification.getByUserId(
        user.id,
        parseInt(limit),
        parseInt(offset)
      );

      console.log(`[NOTIFICATION] Found ${notifications.length} notifications for user ${user.id}`);

      res.json({
        success: true,
        count: notifications.length,
        data: notifications
      });
    } catch (error) {
      console.error(`[NOTIFICATION] Error getting notifications:`, error);
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get unread notifications
  static async getUnreadNotifications(req, res) {
    try {
      const { token } = req.params;

      if (!token) {
        return res.status(400).json({
          success: false,
          message: 'Token is required'
        });
      }

      const user = await User.getByToken(token);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      const notifications = await Notification.getUnreadByUserId(user.id);
      const unreadCount = await Notification.getUnreadCount(user.id);

      res.json({
        success: true,
        unreadCount,
        count: notifications.length,
        data: notifications
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Get unread count
  static async getUnreadCount(req, res) {
    try {
      const { token } = req.params;

      if (!token) {
        return res.status(400).json({
          success: false,
          message: 'Token is required'
        });
      }

      const user = await User.getByToken(token);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      const unreadCount = await Notification.getUnreadCount(user.id);

      res.json({
        success: true,
        unreadCount
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Mark a notification as read
  static async markAsRead(req, res) {
    try {
      const { token, notificationId } = req.params;

      if (!token || !notificationId) {
        return res.status(400).json({
          success: false,
          message: 'Token and notification ID are required'
        });
      }

      const user = await User.getByToken(token);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      const notification = await Notification.markAsRead(notificationId, user.id);

      if (!notification) {
        return res.status(404).json({
          success: false,
          message: 'Notification not found'
        });
      }

      res.json({
        success: true,
        message: 'Notification marked as read',
        data: notification
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Mark all notifications as read
  static async markAllAsRead(req, res) {
    try {
      const { token } = req.params;

      if (!token) {
        return res.status(400).json({
          success: false,
          message: 'Token is required'
        });
      }

      const user = await User.getByToken(token);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      const count = await Notification.markAllAsRead(user.id);

      res.json({
        success: true,
        message: `${count} notifications marked as read`,
        count
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }

  // Delete a notification
  static async deleteNotification(req, res) {
    try {
      const { token, notificationId } = req.params;

      if (!token || !notificationId) {
        return res.status(400).json({
          success: false,
          message: 'Token and notification ID are required'
        });
      }

      const user = await User.getByToken(token);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      const result = await Notification.delete(notificationId, user.id);

      if (!result) {
        return res.status(404).json({
          success: false,
          message: 'Notification not found'
        });
      }

      res.json({
        success: true,
        message: 'Notification deleted successfully'
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: error.message
      });
    }
  }
}

module.exports = NotificationController;
