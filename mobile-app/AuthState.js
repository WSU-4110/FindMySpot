class AuthState {
  render(page, msgId) {
    throw new Error("render() must be implemented");
  }
}

class IdleState extends AuthState {
  render(page, msgId) {
    const el = document.getElementById(msgId);
    if (el) {
      el.classList.add("hidden");
      el.classList.remove("ok", "err");
      el.textContent = "";
    }
  }
}

class LoadingScreen extends AuthState {
  render(page, msgId) {
    page.toast(msgId, "Loading...", "ok");
  }
}

class SuccessState extends AuthState {
  constructor(message) {
    super();
    this.message = message;
  }

  render(page, msgId) {
    page.toast(msgId, this.message, "ok");
  }
}

class ErrorState extends AuthState {
  constructor(message) {
    super();
    this.message = message;
  }

  render(page, msgId) {
    page.toast(msgId, this.message, "err");
  }
}