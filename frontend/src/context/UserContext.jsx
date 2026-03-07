/**
 * UserContext — supports both Firebase Auth and localStorage fallback.
 *
 * When Firebase is configured (VITE_FIREBASE_* env vars filled in):
 *   - login()    → Firebase signInWithEmailAndPassword
 *   - register() → Firebase createUserWithEmailAndPassword
 *   - logout()   → Firebase signOut
 *   - User state persists across browser restarts via Firebase session
 *
 * When Firebase is NOT configured (env vars empty):
 *   - Falls back to the original localStorage approach — app still works
 */
import React, { createContext, useContext, useState, useEffect } from 'react';
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  updateProfile,
} from 'firebase/auth';
import { auth, isConfigured } from '../firebase';

const UserContext = createContext();

/* ── localStorage helpers (fallback) ──────────────────────────────── */
const LS_KEY = 'user';
const loadLocal = () => { try { const s = localStorage.getItem(LS_KEY); return s ? JSON.parse(s) : null; } catch { return null; } };
const saveLocal = (u) => { if (u) localStorage.setItem(LS_KEY, JSON.stringify(u)); else localStorage.removeItem(LS_KEY); };

export function UserProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);   // true while auth state resolves

  /* ── Initialise ───────────────────────────────────────────────── */
  useEffect(() => {
    if (isConfigured && auth) {
      // Firebase: listen for auth state changes
      const unsub = onAuthStateChanged(auth, (firebaseUser) => {
        if (firebaseUser) {
          setUser({
            id: firebaseUser.uid,
            name: firebaseUser.displayName || firebaseUser.email.split('@')[0],
            email: firebaseUser.email,
            role: localStorage.getItem('user_role') || 'student',
            language: localStorage.getItem('user_language') || 'en',
            loginTime: new Date().toISOString(),
          });
        } else {
          setUser(null);
        }
        setLoading(false);
      });
      return unsub;
    } else {
      // Fallback: localStorage
      setUser(loadLocal());
      setLoading(false);
    }
  }, []);

  /* ── Persist role/language locally even for Firebase users ───── */
  useEffect(() => {
    if (!isConfigured) saveLocal(user);
  }, [user]);

  /* ── login ───────────────────────────────────────────────────── */
  const login = async (nameOrEmail, role = 'student', language = 'en', password = '') => {
    if (isConfigured && auth && password) {
      // Firebase email/password login
      const cred = await signInWithEmailAndPassword(auth, nameOrEmail, password);
      const fu = cred.user;
      localStorage.setItem('user_role', role);
      localStorage.setItem('user_language', language);
      const u = {
        id: fu.uid,
        name: fu.displayName || nameOrEmail.split('@')[0],
        email: fu.email,
        role, language,
        loginTime: new Date().toISOString(),
      };
      setUser(u);
      return u;
    }
    // Fallback: localStorage only
    const u = { id: Date.now().toString(), name: nameOrEmail, role, language, loginTime: new Date().toISOString() };
    setUser(u);
    return u;
  };

  /* ── register ────────────────────────────────────────────────── */
  const register = async (name, email, password, role = 'student', language = 'en') => {
    if (isConfigured && auth) {
      const cred = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(cred.user, { displayName: name });
      localStorage.setItem('user_role', role);
      localStorage.setItem('user_language', language);
      const u = { id: cred.user.uid, name, email, role, language, loginTime: new Date().toISOString() };
      setUser(u);
      return u;
    }
    // Fallback
    const u = { id: Date.now().toString(), name, email, role, language, loginTime: new Date().toISOString() };
    setUser(u);
    return u;
  };

  /* ── logout ──────────────────────────────────────────────────── */
  const logout = async () => {
    if (isConfigured && auth) {
      await signOut(auth);
    }
    setUser(null);
    saveLocal(null);
  };

  /* ── updateLanguage ──────────────────────────────────────────── */
  const updateLanguage = (language) => {
    if (user) {
      const u = { ...user, language };
      setUser(u);
      localStorage.setItem('user_language', language);
    }
  };

  return (
    <UserContext.Provider value={{ user, login, register, logout, updateLanguage, loading, isFirebase: isConfigured }}>
      {!loading && children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error('useUser must be used within UserProvider');
  return ctx;
}
