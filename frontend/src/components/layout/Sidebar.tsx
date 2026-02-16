import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  MessageSquare,
  Database,
  FileText,
  Shield,
  Users,
  BarChart,
  Settings,
  LogOut } from
'lucide-react';
import { useAuth } from '../../auth';
export function Sidebar() {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const navItems = [
  {
    icon: LayoutDashboard,
    label: 'Dashboard',
    to: '/dashboard'
  },
  {
    icon: MessageSquare,
    label: 'Ask a Question',
    to: '/ask'
  },
  {
    icon: Database,
    label: 'Retrieval Results',
    to: '/results'
  },
  {
    icon: FileText,
    label: 'Evidence & Citations',
    to: '/answers'
  },
  {
    icon: Users,
    label: 'Clients',
    to: '/clients'
  },
  {
    icon: BarChart,
    label: 'Evaluation',
    to: '/evaluation'
  },
  {
    icon: Settings,
    label: 'Settings',
    to: '/settings'
  }];

  return (
    <div className="w-64 bg-white border-r border-gray-200 h-screen flex flex-col fixed left-0 top-0 z-30">
      <div className="p-6 flex items-center space-x-3">
        <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
          <Shield className="w-5 h-5 text-white" />
        </div>
        <span className="text-lg font-bold text-gray-900">FederatedMed</span>
      </div>

      <nav className="flex-1 px-4 space-y-1 overflow-y-auto">
        {navItems.map((item) =>
        <NavLink
          key={item.to}
          to={item.to}
          className={({ isActive }) => `
              flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-colors
              ${isActive ? 'bg-indigo-50 text-indigo-700' : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'}
            `}>

            <item.icon className="w-5 h-5 mr-3 flex-shrink-0" />
            {item.label}
          </NavLink>
        )}
      </nav>

      <div className="p-4 border-t border-gray-200">
        <button
        className="flex items-center w-full px-3 py-2.5 text-sm font-medium text-red-600 rounded-lg hover:bg-red-50 transition-colors"
        onClick={() => {
          logout();
          navigate('/login', { replace: true });
        }}>
          <LogOut className="w-5 h-5 mr-3" />
          Sign Out
        </button>
      </div>
    </div>);

}