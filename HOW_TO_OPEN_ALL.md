# How to Open All Terminals - Visual Guide

## Step-by-Step Terminal Setup

### 📋 What You Need

You'll need **4 PowerShell windows** open at the same time:
1. **Database Setup** (run once then close)
2. **Backend Server** (keep running)
3. **Mobile App Web Server** (keep running)
4. **Testing/Commands** (optional, for testing)

---

## 🖥️ Opening Windows (Method 1: Using VS Code)

### Method 1A: Built-in Terminal

1. Open VS Code
2. Press `Ctrl + Shift + ` ` (backtick) to open terminal
3. Click the **+** icon to open new terminals
4. Repeat until you have 4 terminals

**Result:**
```
┌─────────────────────────────────────────────────────┐
│ VS Code                                             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Backend] [MobileApp] [Database] [Testing] [+]    │
│                                                     │
│  Terminal output here...                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🖥️ Opening Windows (Method 2: Manual PowerShell)

### Step 1: Open First PowerShell

**Click Start → Type "PowerShell" → Click Windows PowerShell**

or use keyboard shortcut:

```
Windows Key + X → Select "Windows PowerShell"
```

---

### Step 2: Run Database Setup

**In Terminal 1:**

```powershell
cd C:\Users\asmit\FindMySpot
.\setup-database.ps1
```

Wait for it to complete, then close this terminal.

---

### Step 3: Open Second PowerShell (Backend)

**Click Start → Type "PowerShell" → Click Windows PowerShell**

**In Terminal 2:**

```powershell
cd C:\Users\asmit\FindMySpot\backend
npm run dev
```

Leave this running (don't close).

---

### Step 4: Open Third PowerShell (Mobile App)

**Click Start → Type "PowerShell" → Click Windows PowerShell**

**In Terminal 3:**

```powershell
cd C:\Users\asmit\FindMySpot\mobile-app
python -m http.server 8080
```

Leave this running (don't close).

---

### Step 5: Open Fourth PowerShell (Testing)

**Click Start → Type "PowerShell" → Click Windows PowerShell**

**In Terminal 4:**

```powershell
cd C:\Users\asmit\FindMySpot
python test_system.py
```

---

## Visual Layout After Setup

```
┌──────────────────────────┬──────────────────────────┐
│   Terminal 2: Backend    │  Terminal 3: Web App     │
│   Port: 3000             │  Port: 8080              │
│                          │                          │
│ $ npm run dev            │ $ python -m http.server  │
│ > (running...)           │ > Serving HTTP on...     │
│                          │                          │
│ http://localhost:3000    │ http://localhost:8080    │
│                          │                          │
└──────────────────────────┴──────────────────────────┘

┌──────────────────────────┬──────────────────────────┐
│  Terminal 1: Database    │  Terminal 4: Tests       │
│  (Setup Complete)        │  (Run Tests)             │
│                          │                          │
│ $ .\setup-database.ps1   │ $ python test_system.py  │
│ > ✓ Setup complete!      │ > Running tests...       │
│                          │                          │
│ ✅ CLOSE THIS            │ (Can close after)        │
│                          │                          │
└──────────────────────────┴──────────────────────────┘
```

---

## ✅ Verification Checklist

After opening all terminals, verify:

### Terminal 1: Database Setup
```
✓ Step 1: Running initial migration
✓ CREATE TABLE
✓ CREATE INDEX
✓ Step 2: Running license plate detection migration
✓ Database setup completed successfully!
```

### Terminal 2: Backend Server (Port 3000)
```
✓ npm notice
✓ npm WARN
✓ FindMySpot API server running on port 3000
✓ Visit http://localhost:3000 for API documentation
```

### Terminal 3: Mobile App (Port 8080)
```
✓ Serving HTTP on 0.0.0.0 port 8080
✓ [IP Address] - - [Date Time] "GET / HTTP/1.1" 200 -
```

### Terminal 4: Tests (Optional)
```
✓ API Server is running
✓ User registered
✓ Vehicle registered
✓ Retrieved vehicles
✓ Detection recorded
✓ All Tests Completed Successfully!
```

---

## 🌐 Test Access

After all terminals are running:

| URL | Purpose |
|-----|---------|
| http://localhost:3000 | Backend API (see available endpoints) |
| http://localhost:8080 | Mobile App (Sign up & use app) |

---

## 🎯 Next Steps After Setup

1. **Open mobile app** → http://localhost:8080
2. **Create account** → Sign up page
3. **Register vehicle** → Go to "My Vehicles"
4. **See notifications** → Go to "Notifications" page
5. **Test detection** → Run `python ai-service/detection_service.py` in Terminal 4

---

## ⚠️ Important Notes

- **Don't close Terminal 2 or 3** - they need to keep running
- **Terminal 1** can be closed after database setup completes
- **Terminal 4** is for commands/testing, you can open/close as needed
- **All 4 must be open** for the system to work properly

---

## 🆘 Something Not Working?

**Check WINDOWS_SETUP_GUIDE.md** in the FindMySpot folder for:
- Common errors and solutions
- Port troubleshooting
- Service errors
- Database connection issues

---

**You're all set! Start with the mobile app at http://localhost:8080 🚀**
