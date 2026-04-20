const Module = require('module');

function loadWithMocks(targetModulePath, mocks) {
  const originalLoad = Module._load;

  Module._load = function patchedLoad(request, parent, isMain) {
    if (Object.prototype.hasOwnProperty.call(mocks, request)) {
      return mocks[request];
    }
    return originalLoad(request, parent, isMain);
  };

  try {
    delete require.cache[require.resolve(targetModulePath)];
    return require(targetModulePath);
  } finally {
    Module._load = originalLoad;
  }
}

function createMockRes() {
  return {
    statusCode: 200,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.body = payload;
      return this;
    },
  };
}

module.exports = {
  loadWithMocks,
  createMockRes,
};