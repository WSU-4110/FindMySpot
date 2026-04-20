const test = require('node:test');
const assert = require('node:assert/strict');

const { loadWithMocks, createMockRes } = require('./backend_test_utils');

const CONTROLLER_PATH = '../backend/controllers/parkingController';

async function withTimerMocks(fn) {
  const originalSetTimeout = global.setTimeout;
  const originalClearTimeout = global.clearTimeout;
  global.setTimeout = () => ({ mocked: true });
  global.clearTimeout = () => {};
  try {
    return await fn();
  } finally {
    global.setTimeout = originalSetTimeout;
    global.clearTimeout = originalClearTimeout;
  }
}

test('ParkingController.checkIn returns 400 for missing required fields', async () => {
  const mockParkingModel = {
    ParkingSpot: {},
    Vehicle: {},
    ParkingSession: { create: async () => ({}) },
    SecurityFlag: {},
  };

  const ParkingController = loadWithMocks(CONTROLLER_PATH, {
    '../models/parking': mockParkingModel,
    '../models/user': { User: {} },
  });

  const req = { body: { vehiclePlate: '', floor: 2, lot: 1 } };
  const res = createMockRes();

  await ParkingController.checkIn(req, res);

  assert.equal(res.statusCode, 400);
  assert.equal(res.body.success, false);
  assert.match(res.body.message, /required/i);
});

test('ParkingController.checkIn creates a session and returns success', async () => {
  const fakeSession = {
    id: 21,
    vehicle_plate: 'ABC123',
    floor: 2,
    lot: 5,
    check_in_time: new Date().toISOString(),
  };

  const mockParkingModel = {
    ParkingSpot: {
      getOccupancyStats: async () => ({ total: 25, occupied: 1, available: 24, byFloor: {} }),
    },
    Vehicle: {},
    ParkingSession: {
      create: async (plate, floor, lot) => {
        assert.equal(plate, 'ABC123');
        assert.equal(floor, 2);
        assert.equal(lot, 5);
        return fakeSession;
      },
    },
    SecurityFlag: {},
  };

  const ParkingController = loadWithMocks(CONTROLLER_PATH, {
    '../models/parking': mockParkingModel,
    '../models/user': { User: {} },
  });

  const req = { body: { vehiclePlate: 'ABC123', floor: 2, lot: 5 } };
  const res = createMockRes();

  await withTimerMocks(async () => {
    await ParkingController.checkIn(req, res);
  });

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.success, true);
  assert.equal(res.body.data.id, 21);
});

test('ParkingController.checkOut returns 400 when vehiclePlate is missing', async () => {
  const mockParkingModel = {
    ParkingSpot: {},
    Vehicle: {},
    ParkingSession: { checkout: async () => ({}) },
    SecurityFlag: {},
  };

  const ParkingController = loadWithMocks(CONTROLLER_PATH, {
    '../models/parking': mockParkingModel,
    '../models/user': { User: {} },
  });

  const req = { body: {} };
  const res = createMockRes();

  await ParkingController.checkOut(req, res);

  assert.equal(res.statusCode, 400);
  assert.equal(res.body.success, false);
});

test('ParkingController.getOccupancyStats falls back to stale data when live fetch fails', async () => {
  let callCount = 0;
  const mockParkingModel = {
    ParkingSpot: {
      getOccupancyStats: async () => {
        callCount += 1;
        if (callCount === 1) {
          return { total: 25, occupied: 4, available: 21, byFloor: { 1: { total: 5, occupied: 1, available: 4 } } };
        }
        throw new Error('db unavailable');
      },
    },
    Vehicle: {},
    ParkingSession: {},
    SecurityFlag: {},
  };

  const ParkingController = loadWithMocks(CONTROLLER_PATH, {
    '../models/parking': mockParkingModel,
    '../models/user': { User: {} },
  });

  const firstRes = createMockRes();
  await ParkingController.getOccupancyStats({}, firstRes);
  assert.equal(firstRes.statusCode, 200);
  assert.equal(firstRes.body.stale, false);

  const secondRes = createMockRes();
  await ParkingController.getOccupancyStats({}, secondRes);
  assert.equal(secondRes.statusCode, 200);
  assert.equal(secondRes.body.success, true);
  assert.equal(secondRes.body.stale, true);
  assert.match(secondRes.body.message, /most recent known occupancy data/i);
});