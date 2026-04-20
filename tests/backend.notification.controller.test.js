const test = require('node:test');
const assert = require('node:assert/strict');

const { loadWithMocks, createMockRes } = require('./backend_test_utils');

const CONTROLLER_PATH = '../backend/controllers/notificationController';

test('NotificationController.getUserNotifications returns 400 when token is missing', async () => {
  const NotificationController = loadWithMocks(CONTROLLER_PATH, {
    '../models/notification': {},
    '../models/user': { User: { getByToken: async () => null } },
  });

  const req = { params: {}, query: {} };
  const res = createMockRes();

  await NotificationController.getUserNotifications(req, res);

  assert.equal(res.statusCode, 400);
  assert.equal(res.body.success, false);
  assert.match(res.body.message, /token is required/i);
});

test('NotificationController.getUserNotifications returns 401 for invalid token', async () => {
  const NotificationController = loadWithMocks(CONTROLLER_PATH, {
    '../models/notification': { getByUserId: async () => [] },
    '../models/user': { User: { getByToken: async () => null } },
  });

  const req = { params: { token: 'bad-token' }, query: {} };
  const res = createMockRes();

  await NotificationController.getUserNotifications(req, res);

  assert.equal(res.statusCode, 401);
  assert.equal(res.body.success, false);
});

test('NotificationController.getUserNotifications returns notifications for valid token', async () => {
  const notifications = [
    { id: 1, title: 'Detected', message: 'Vehicle detected' },
    { id: 2, title: 'Reminder', message: 'Session nearing timeout' },
  ];

  const NotificationController = loadWithMocks(CONTROLLER_PATH, {
    '../models/notification': {
      getByUserId: async (userId, limit, offset) => {
        assert.equal(userId, 56);
        assert.equal(limit, 10);
        assert.equal(offset, 0);
        return notifications;
      },
    },
    '../models/user': {
      User: {
        getByToken: async (token) => {
          assert.equal(token, 'good-token');
          return { id: 56 };
        },
      },
    },
  });

  const req = { params: { token: 'good-token' }, query: { limit: '10', offset: '0' } };
  const res = createMockRes();

  await NotificationController.getUserNotifications(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.success, true);
  assert.equal(res.body.count, 2);
  assert.deepEqual(res.body.data, notifications);
});

test('NotificationController.markAsRead returns 404 when notification is missing', async () => {
  const NotificationController = loadWithMocks(CONTROLLER_PATH, {
    '../models/notification': {
      markAsRead: async () => null,
    },
    '../models/user': {
      User: {
        getByToken: async () => ({ id: 56 }),
      },
    },
  });

  const req = { params: { token: 'good-token', notificationId: '999' } };
  const res = createMockRes();

  await NotificationController.markAsRead(req, res);

  assert.equal(res.statusCode, 404);
  assert.equal(res.body.success, false);
  assert.match(res.body.message, /not found/i);
});

test('NotificationController.markAllAsRead returns count for valid user', async () => {
  const NotificationController = loadWithMocks(CONTROLLER_PATH, {
    '../models/notification': {
      markAllAsRead: async (userId) => {
        assert.equal(userId, 56);
        return 3;
      },
    },
    '../models/user': {
      User: {
        getByToken: async () => ({ id: 56 }),
      },
    },
  });

  const req = { params: { token: 'good-token' } };
  const res = createMockRes();

  await NotificationController.markAllAsRead(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.success, true);
  assert.equal(res.body.count, 3);
});