# 🫀 ECG Smartwatch App - Complete Setup Guide

## 🎯 **WHAT WE'RE BUILDING**

A **complete React Native app** that:
- ✅ Connects to **any smartwatch** (Apple Watch, Samsung, Fitbit, Amazfit)
- ✅ Uses your **MobileNet v1 model** for real-time ECG analysis  
- ✅ Generates **simple health reports** in plain English
- ✅ Works as a **copy-paste module** for future projects
- ✅ Provides **emergency alerts** for dangerous heart rhythms

---

## 📋 **STEP-BY-STEP SETUP (Follow Exactly)**

### **PHASE 1: CONVERT YOUR MODEL (Do This First!)**

1. **Copy your trained model files** from `mod/` folder to a new location
2. **Run the model conversion script**:

```bash
cd "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis\mod"
python ../ecg_react_native_feature/STEP_1A_Convert_Model.py
```

3. **Expected result**: You should get a file called `mobilenet_v1_ecg_model.onnx` (about 4-5MB)

### **PHASE 2: INSTALL DEVELOPMENT TOOLS**

**2.1 Install Node.js**
- Go to https://nodejs.org/
- Download and install **Node.js 18+**
- Verify: Open PowerShell and run `node --version`

**2.2 Install Java Development Kit**
- Go to https://adoptium.net/
- Download and install **JDK 11**
- Add to environment variables: `JAVA_HOME = C:\Program Files\Eclipse Adoptium\jdk-11.x.x-hotspot`

**2.3 Install Android Studio**
- Go to https://developer.android.com/studio
- Download and install **Android Studio**
- During installation, install: Android SDK, Android SDK Platform, Android Virtual Device

**2.4 Setup Android Environment**
Add these to your Windows Environment Variables:
```
ANDROID_HOME = C:\Users\%USERNAME%\AppData\Local\Android\Sdk
```
Add to PATH:
```
%ANDROID_HOME%\platform-tools
%ANDROID_HOME%\emulator
%ANDROID_HOME%\tools
%ANDROID_HOME%\tools\bin
```

**2.5 Install React Native CLI**
```bash
npm install -g react-native-cli
```

### **PHASE 3: CREATE YOUR APP PROJECT**

Open PowerShell and run:

```powershell
# Navigate to your project directory
cd "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis"

# Create new React Native project
npx react-native init ECGSmartwatchApp --template react-native-template-typescript

# Navigate to project directory
cd ECGSmartwatchApp

# Install all dependencies for our ECG feature
npm install react-native-ble-plx@^3.0.0
npm install @react-native-async-storage/async-storage@^1.19.0
npm install react-native-fs@^2.20.0
npm install react-native-permissions@^3.8.0
npm install react-native-svg@^13.9.0
npm install react-native-vector-icons@^9.2.0
npm install react-native-chart-kit@^6.12.0
npm install react-native-linear-gradient@^2.8.0
npm install react-native-sound@^0.11.2
npm install react-native-haptic-feedback@^1.14.0

# For TensorFlow Lite (alternative to ONNX)
npm install react-native-tensorflow-lite@^1.0.0
```

### **PHASE 4: COPY OUR ECG FEATURE MODULE**

```powershell
# Copy the entire ECG feature folder to your new project
xcopy "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis\ecg_react_native_feature\src" "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis\ECGSmartwatchApp\src" /E /I

# Copy assets
xcopy "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis\ecg_react_native_feature\assets" "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis\ECGSmartwatchApp\assets" /E /I

# Copy your converted model file
copy "mobilenet_v1_ecg_model.onnx" "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis\ECGSmartwatchApp\assets\ml_models\"
```

### **PHASE 5: CONFIGURE ANDROID PERMISSIONS**

Edit `android/app/src/main/AndroidManifest.xml` and add these permissions:

```xml
<uses-permission android:name="android.permission.BLUETOOTH" />
<uses-permission android:name="android.permission.BLUETOOTH_ADMIN" />
<uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
<uses-permission android:name="android.permission.BLUETOOTH_SCAN" />
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
```

### **PHASE 6: TEST THE BASE APP**

```powershell
# Start Metro bundler (keep this running)
npm start

# In a new PowerShell window, run on Android
npx react-native run-android
```

**Expected result**: You should see a basic React Native app running on your Android emulator or device.

---

## 🚀 **INTEGRATION STEPS (After Base App Works)**

### **Step 1: Add ECG Feature to Main App**

Edit `App.tsx` in your project root:

```typescript
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

// Import our ECG feature screens
import ECGHomeScreen from './src/features/ecg_smartwatch_monitor/presentation/screens/ECGHomeScreen';
import LiveMeasurementScreen from './src/features/ecg_smartwatch_monitor/presentation/screens/LiveMeasurementScreen';
import HealthReportScreen from './src/features/ecg_smartwatch_monitor/presentation/screens/HealthReportScreen';
import HistoryScreen from './src/features/ecg_smartwatch_monitor/presentation/screens/HistoryScreen';

const Stack = createStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="ECGHome">
        <Stack.Screen 
          name="ECGHome" 
          component={ECGHomeScreen}
          options={{ title: 'ECG Monitor' }}
        />
        <Stack.Screen 
          name="LiveMeasurement" 
          component={LiveMeasurementScreen}
          options={{ title: 'ECG Measurement' }}
        />
        <Stack.Screen 
          name="HealthReport" 
          component={HealthReportScreen}
          options={{ title: 'Health Report' }}
        />
        <Stack.Screen 
          name="History" 
          component={HistoryScreen}
          options={{ title: 'ECG History' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
```

### **Step 2: Install Navigation Dependencies**

```bash
npm install @react-navigation/native@^6.1.7
npm install @react-navigation/stack@^6.3.17
npm install react-native-screens@^3.22.0
npm install react-native-safe-area-context@^4.7.1
```

### **Step 3: Test ECG Feature Integration**

```bash
# Rebuild the app with our ECG feature
npx react-native run-android
```

---

## 📱 **TESTING CHECKLIST**

After setup, you should be able to:

- [ ] ✅ App starts without crashes
- [ ] ✅ See "ECG Monitor" home screen
- [ ] ✅ Tap "Start Scanning" button
- [ ] ✅ See Bluetooth permission request
- [ ] ✅ Navigate between screens
- [ ] ✅ See placeholder UI for all features

---

## 🎯 **NEXT PHASES (After Basic Setup Works)**

### **Week 1**: Model Integration & Real-time Analysis
### **Week 2**: Bluetooth Connection & Data Streaming  
### **Week 3**: Health Report Generation
### **Week 4**: UI/UX Polish & Testing
### **Week 5-8**: Advanced Features & Deployment

---

## ⚠️ **TROUBLESHOOTING**

**If Android build fails:**
```bash
cd android
./gradlew clean
cd ..
npx react-native run-android
```

**If Metro bundler issues:**
```bash
npx react-native start --reset-cache
```

**If permission errors:**
- Make sure Android SDK is properly installed
- Check ANDROID_HOME environment variable
- Run Android Studio and install missing SDK components

**If Bluetooth doesn't work:**
- Test on real Android device (not emulator)
- Check location permissions are granted
- Ensure target Android version is 6.0+ (API 23+)

---

## 🎉 **SUCCESS CRITERIA**

When setup is complete, you should have:
- ✅ Working React Native app
- ✅ ECG feature integrated
- ✅ All dependencies installed
- ✅ App runs on Android device/emulator
- ✅ Basic navigation works
- ✅ Ready for Phase 2 development

**Reply "SETUP COMPLETE" when you've got the basic app running!**
