# STEP 1B: SETUP REACT NATIVE DEVELOPMENT ENVIRONMENT

## What you need to install on your Windows machine:

### 1. Install Node.js (Required)
```bash
# Download and install Node.js 18+ from: https://nodejs.org/
# Verify installation:
node --version  # Should show v18.x.x or higher
npm --version   # Should show 9.x.x or higher
```

### 2. Install Java Development Kit (Required for Android)
```bash
# Download and install JDK 11 from: https://adoptium.net/
# Add to environment variables:
# JAVA_HOME = C:\Program Files\Eclipse Adoptium\jdk-11.x.x-hotspot
```

### 3. Install Android Studio (Required)
```bash
# Download from: https://developer.android.com/studio
# During installation, make sure to install:
# - Android SDK
# - Android SDK Platform
# - Android Virtual Device
```

### 4. Install React Native CLI
```bash
npm install -g react-native-cli
npx react-native --version  # Verify installation
```

### 5. Setup Android Environment Variables
```bash
# Add these to your Windows Environment Variables:
# ANDROID_HOME = C:\Users\%USERNAME%\AppData\Local\Android\Sdk
# Add to PATH:
# %ANDROID_HOME%\platform-tools
# %ANDROID_HOME%\emulator
# %ANDROID_HOME%\tools
# %ANDROID_HOME%\tools\bin
```

## STEP 1C: CREATE YOUR ECG APP PROJECT

### Open PowerShell and run these commands:

```powershell
# Navigate to your project directory
cd "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis"

# Create new React Native project
npx react-native init ECGSmartwatchApp --template react-native-template-typescript

# Navigate to project directory
cd ECGSmartwatchApp

# Copy our pre-built ECG feature module
# (We'll do this after setting up the base project)
```

## STEP 1D: TEST BASIC APP SETUP

```powershell
# Start Metro bundler
npm start

# In a new PowerShell window, run on Android:
npx react-native run-android
```

## 🎯 WHAT TO DO RIGHT NOW:

1. **FIRST**: Run the Python script to convert your model (STEP_1A_Convert_Model.py)
2. **SECOND**: Install all the development tools above
3. **THIRD**: Create the React Native project
4. **FOURTH**: Come back here and I'll show you how to integrate our ECG feature

## ⚠️ COMMON ISSUES & SOLUTIONS:

**If Android Studio installation fails:**
- Make sure you have enough disk space (8GB+)
- Run as administrator
- Disable antivirus temporarily during installation

**If React Native project creation fails:**
- Clear npm cache: `npm cache clean --force`
- Update npm: `npm install -g npm@latest`
- Try again with: `npx --clear-cache react-native init ECGSmartwatchApp`

**If Android emulator won't start:**
- Enable Hyper-V in Windows features
- Make sure Intel HAXM is installed
- Allocate at least 4GB RAM to emulator

## 📱 NEXT PHASE:
Once you complete these steps, we'll:
1. Copy our ECG feature module into your new project
2. Install all dependencies
3. Set up the MobileNet model integration
4. Test Bluetooth connectivity
5. Create the first working prototype

**Reply "DONE STEP 1" when you've completed the environment setup!**
