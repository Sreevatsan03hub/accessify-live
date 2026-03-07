// Firebase client configuration for Accessify
// ────────────────────────────────────────────────────────────────────
// HOW TO FILL THIS IN:
// 1. Go to https://console.firebase.google.com
// 2. Open your project → Project Settings (⚙) → Your apps → Web app
// 3. Copy the firebaseConfig object and paste here
// 4. Enable Authentication → Email/Password in the Firebase Console
// 5. Enable Firestore Database in the Firebase Console
// ────────────────────────────────────────────────────────────────────

import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';

const firebaseConfig = {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY || '',
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || '',
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || '',
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || '',
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || '',
    appId: import.meta.env.VITE_FIREBASE_APP_ID || '',
};

// Only initialise if credentials are provided
const isConfigured = Boolean(firebaseConfig.apiKey && firebaseConfig.projectId);

let app = null;
let auth = null;
let db = null;
let fbStorage = null;

if (isConfigured) {
    app = initializeApp(firebaseConfig);
    auth = getAuth(app);
    db = getFirestore(app);
    fbStorage = getStorage(app);
    console.log('🔥 Firebase initialised with API Key:', firebaseConfig.apiKey);
} else {
    console.info(
        '📁 Firebase not configured — using local storage fallback.\n' +
        'Add VITE_FIREBASE_* variables to frontend/.env.local to enable Firebase.'
    );
}

export { auth, db, fbStorage, isConfigured };
export default app;
