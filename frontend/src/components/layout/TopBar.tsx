import React from 'react';
import { Bell, Search, User } from 'lucide-react';
import { Input } from '../ui/Input';
import { useAuth } from '../../auth';
export function TopBar() {
  const { user, isAuthenticated } = useAuth();
  const displayName = isAuthenticated ? user?.email || 'User' : 'Guest';
  return (
    <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 fixed top-0 right-0 left-64 z-20">
      <div className="flex items-center w-96">
        <Input
          placeholder="Search..."
          icon={<Search className="w-4 h-4" />}
          className="bg-gray-50 border-transparent focus:bg-white" />

      </div>

      <div className="flex items-center space-x-4">
        <div className="flex items-center px-3 py-1.5 bg-indigo-50 rounded-full border border-indigo-100">
          <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
          <span className="text-xs font-medium text-indigo-700">
            Consortium A
          </span>
        </div>

        <button className="relative p-2 text-gray-500 hover:bg-gray-100 rounded-full transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full border-2 border-white"></span>
        </button>

        <div className="flex items-center space-x-3 pl-4 border-l border-gray-200">
          <div className="text-right hidden sm:block">
            <p className="text-sm font-medium text-gray-900">{displayName}</p>
            <p className="text-xs text-gray-500">
              {isAuthenticated ? 'Signed in' : 'Not signed in'}
            </p>
          </div>
          <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center border border-indigo-200">
            <User className="w-5 h-5 text-indigo-600" />
          </div>
        </div>
      </div>
    </header>);

}