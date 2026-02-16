import React, { useEffect, useState, Children } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Gauge } from '../components/ui/Gauge';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import {
  MessageSquare,
  Activity,
  Database,
  Clock,
  ArrowRight } from
'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer } from
'recharts';
const healthData = [
{
  time: '00:00',
  latency: 120,
  queries: 45
},
{
  time: '04:00',
  latency: 132,
  queries: 30
},
{
  time: '08:00',
  latency: 145,
  queries: 85
},
{
  time: '12:00',
  latency: 180,
  queries: 120
},
{
  time: '16:00',
  latency: 160,
  queries: 95
},
{
  time: '20:00',
  latency: 140,
  queries: 60
},
{
  time: '23:59',
  latency: 125,
  queries: 40
}];

export function DashboardPage() {
  const [isLoading, setIsLoading] = useState(true);
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 1000);
    return () => clearTimeout(timer);
  }, []);
  const container = {
    hidden: {
      opacity: 0
    },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  const item = {
    hidden: {
      opacity: 0,
      y: 20
    },
    show: {
      opacity: 1,
      y: 0
    }
  };
  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6">

      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <span className="text-sm text-gray-500">Last updated: Just now</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Federated Query Card - Main Action */}
        <motion.div variants={item} className="lg:col-span-2">
          <Card className="h-full bg-gradient-to-r from-indigo-600 to-indigo-800 text-white border-none relative overflow-hidden">
            <div className="relative z-10 p-2">
              <h2 className="text-2xl font-bold mb-2">
                Start a Federated Query
              </h2>
              <p className="text-indigo-100 mb-6 max-w-lg">
                Securely query medical knowledge across 3 hospital nodes without
                accessing raw patient data.
              </p>
              <Link to="/ask">
                <Button
                  variant="secondary"
                  size="lg"
                  rightIcon={<ArrowRight className="w-4 h-4" />}>

                  Ask a Question
                </Button>
              </Link>
            </div>
            <div className="absolute right-0 bottom-0 opacity-10 transform translate-x-1/4 translate-y-1/4">
              <Database className="w-64 h-64" />
            </div>
          </Card>
        </motion.div>

        {/* Privacy Budget Card */}
        <motion.div variants={item}>
          <Card className="h-full flex flex-col items-center justify-center">
            <h3 className="text-lg font-medium text-gray-900 mb-4 self-start w-full">
              Privacy Budget
            </h3>
            {isLoading ?
            <Skeleton circle width={120} height={120} /> :

            <Gauge value={0.3} label="Monthly Usage" />
            }
            <div className="mt-4 text-center">
              <p className="text-sm text-gray-500">Remaining: ε = 0.7</p>
              <p className="text-xs text-gray-400">Resets in 12 days</p>
            </div>
          </Card>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Client Status */}
        <motion.div variants={item} className="lg:col-span-1">
          <Card>
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Client Status
            </h3>
            <div className="space-y-4">
              {[
              {
                name: 'Client 0 (General)',
                status: 'online',
                docs: '8,420'
              },
              {
                name: 'Client 1 (Memorial)',
                status: 'online',
                docs: '12,150'
              },
              {
                name: 'Client 2 (Research)',
                status: 'offline',
                docs: '5,300'
              }].
              map((client, idx) =>
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">

                  <div className="flex items-center">
                    <div
                    className={`w-2 h-2 rounded-full mr-3 ${client.status === 'online' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />

                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {client.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {client.docs} documents
                      </p>
                    </div>
                  </div>
                  <Badge
                  variant={client.status === 'online' ? 'success' : 'neutral'}>

                    {client.status}
                  </Badge>
                </div>
              )}
            </div>
          </Card>
        </motion.div>

        {/* Recent Questions */}
        <motion.div variants={item} className="lg:col-span-1">
          <Card>
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Recent Questions
            </h3>
            <div className="space-y-4">
              {[
              {
                q: 'ACE inhibitors efficacy in heart failure...',
                time: '2m ago',
                status: 'completed'
              },
              {
                q: 'Metformin outcomes for Type 2...',
                time: '15m ago',
                status: 'completed'
              },
              {
                q: 'Pediatric asthma protocols...',
                time: '1h ago',
                status: 'processing'
              }].
              map((item, idx) =>
              <div
                key={idx}
                className="flex items-start justify-between border-b border-gray-100 last:border-0 pb-3 last:pb-0">

                  <div className="flex items-start">
                    <MessageSquare className="w-4 h-4 text-gray-400 mt-1 mr-3 flex-shrink-0" />
                    <div>
                      <p className="text-sm text-gray-900 line-clamp-1">
                        {item.q}
                      </p>
                      <p className="text-xs text-gray-500 flex items-center mt-1">
                        <Clock className="w-3 h-3 mr-1" /> {item.time}
                      </p>
                    </div>
                  </div>
                  <Badge
                  variant={
                  item.status === 'completed' ? 'success' : 'warning'
                  }
                  size="sm">

                    {item.status}
                  </Badge>
                </div>
              )}
            </div>
            <div className="mt-4 pt-2 border-t border-gray-100">
              <Link
                to="/results"
                className="text-sm text-indigo-600 hover:text-indigo-700 font-medium flex items-center justify-center">

                View all history <ArrowRight className="w-4 h-4 ml-1" />
              </Link>
            </div>
          </Card>
        </motion.div>

        {/* System Health */}
        <motion.div variants={item} className="lg:col-span-1">
          <Card>
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              System Latency
            </h3>
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={healthData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    stroke="#E5E7EB" />

                  <XAxis
                    dataKey="time"
                    tick={{
                      fontSize: 10
                    }}
                    tickLine={false}
                    axisLine={false} />

                  <YAxis
                    tick={{
                      fontSize: 10
                    }}
                    tickLine={false}
                    axisLine={false} />

                  <Tooltip
                    contentStyle={{
                      borderRadius: '8px',
                      border: 'none',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                    itemStyle={{
                      fontSize: '12px'
                    }} />

                  <Line
                    type="monotone"
                    dataKey="latency"
                    stroke="#4F46E5"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{
                      r: 4
                    }} />

                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-2">
              <span>Avg: 142ms</span>
              <span>Peak: 180ms</span>
            </div>
          </Card>
        </motion.div>
      </div>
    </motion.div>);

}