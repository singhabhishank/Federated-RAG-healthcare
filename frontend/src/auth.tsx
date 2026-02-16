import React, { createContext, useContext, useEffect, useState } from 'react';

interface User {
  email: string;
}

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem('authUser');
      if (raw) {
        setUser(JSON.parse(raw) as User);
      }
    } catch {
      setUser(null);
    }
  }, []);

  const login = (email: string) => {
    const u = { email };
    setUser(u);
    try {
      window.localStorage.setItem('authUser', JSON.stringify(u));
    } catch {
      // ignore
    }
  };

  const logout = () => {
    setUser(null);
    try {
      window.localStorage.removeItem('authUser');
      // Optional: clear query history on logout
      // window.localStorage.removeItem('federatedQueryHistory');
    } catch {
      // ignore
    }
  };

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return ctx;
}

