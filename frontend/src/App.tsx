import React from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate } from
'react-router-dom';
import { AppLayout } from './components/layout/AppLayout';
import { LandingPage } from './pages/LandingPage';
import { LoginPage } from './pages/LoginPage';
import { SignupPage } from './pages/SignupPage';
import { ForgotPasswordPage } from './pages/ForgotPasswordPage';
import { DashboardPage } from './pages/DashboardPage';
import { AskQuestionPage } from './pages/AskQuestionPage';
import { RetrievalResultsPage } from './pages/RetrievalResultsPage';
import { AnswerOutputPage } from './pages/AnswerOutputPage';
import { ClientsPage } from './pages/ClientsPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { SettingsPage } from './pages/SettingsPage';
import { AuthProvider, useAuth } from './auth';

// Wrapper for protected routes
function ProtectedRoute({ children }: {children: React.ReactNode;}) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return <AppLayout>{children}</AppLayout>;
}
export function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
        {/* Public Routes */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />

        {/* Protected Routes */}
        <Route
          path="/dashboard"
          element={
          <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          } />

        <Route
          path="/ask"
          element={
          <ProtectedRoute>
              <AskQuestionPage />
            </ProtectedRoute>
          } />

        <Route
          path="/results"
          element={
          <ProtectedRoute>
              <RetrievalResultsPage />
            </ProtectedRoute>
          } />

        <Route
          path="/answers"
          element={
          <ProtectedRoute>
              <AnswerOutputPage />
            </ProtectedRoute>
          } />

        <Route
          path="/clients"
          element={
          <ProtectedRoute>
              <ClientsPage />
            </ProtectedRoute>
          } />

        <Route
          path="/evaluation"
          element={
          <ProtectedRoute>
              <EvaluationPage />
            </ProtectedRoute>
          } />

        <Route
          path="/settings"
          element={
          <ProtectedRoute>
              <SettingsPage />
            </ProtectedRoute>
          } />


        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </Router>
    </AuthProvider>);

}