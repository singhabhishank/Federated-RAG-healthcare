import React from 'react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { ToastProvider } from '../ui/Toast';
interface AppLayoutProps {
  children: React.ReactNode;
}
export function AppLayout({ children }: AppLayoutProps) {
  return (
    <ToastProvider>
      <div className="min-h-screen bg-gray-50">
        <Sidebar />
        <TopBar />
        <main className="pl-64 pt-16 min-h-screen">
          <div className="max-w-7xl mx-auto p-8">{children}</div>
        </main>
      </div>
    </ToastProvider>);

}