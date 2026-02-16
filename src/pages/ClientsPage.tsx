import React from 'react';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Database, Activity, Server, RefreshCw } from 'lucide-react';
export function ClientsPage() {
  const clients = [
  {
    id: 0,
    name: 'General Hospital',
    status: 'online',
    docs: 8420,
    latency: '45ms',
    version: 'v2.1.0'
  },
  {
    id: 1,
    name: 'Memorial Center',
    status: 'online',
    docs: 12150,
    latency: '52ms',
    version: 'v2.1.0'
  },
  {
    id: 2,
    name: 'Research Institute',
    status: 'offline',
    docs: 5300,
    latency: '-',
    version: 'v2.0.9'
  }];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Connected Clients</h1>
        <Button leftIcon={<RefreshCw className="w-4 h-4" />}>
          Refresh Status
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {clients.map((client) =>
        <Card key={client.id} className="relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4">
              <Badge
              variant={client.status === 'online' ? 'success' : 'neutral'}>

                {client.status}
              </Badge>
            </div>

            <div className="flex items-center mb-6">
              <div className="w-12 h-12 bg-indigo-50 rounded-lg flex items-center justify-center mr-4">
                <Server className="w-6 h-6 text-indigo-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-gray-900">
                  {client.name}
                </h3>
                <p className="text-sm text-gray-500">Client ID: {client.id}</p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center text-sm text-gray-600">
                  <Database className="w-4 h-4 mr-2" />
                  Indexed Documents
                </div>
                <span className="font-bold text-gray-900">
                  {client.docs.toLocaleString()}
                </span>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center text-sm text-gray-600">
                  <Activity className="w-4 h-4 mr-2" />
                  Avg Latency
                </div>
                <span className="font-mono text-gray-900">
                  {client.latency}
                </span>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-gray-100 flex justify-between items-center">
              <span className="text-xs text-gray-400">
                Version: {client.version}
              </span>
              <Button variant="ghost" size="sm">
                View Logs
              </Button>
            </div>
          </Card>
        )}
      </div>
    </div>);

}