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
  Database,
  Clock,
  ArrowRight,
  Shield,
  BarChart3,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer } from
'recharts';
import { getClients, getPrivacyBudget, type ClientInfo } from '../api';

const activitySample = [
  { time: '00:00', queries: 12 },
  { time: '04:00', queries: 8 },
  { time: '08:00', queries: 24 },
  { time: '12:00', queries: 31 },
  { time: '16:00', queries: 22 },
  { time: '20:00', queries: 18 },
  { time: '23:59', queries: 14 },
];

export function DashboardPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [recentQuestions, setRecentQuestions] = useState<{ q: string; time: string; status: string }[]>([]);
  const [clients, setClients] = useState<ClientInfo[]>([]);
  const [clientsError, setClientsError] = useState<string | null>(null);
  const [budgetUsed, setBudgetUsed] = useState<number | null>(null);
  const [budgetCap, setBudgetCap] = useState<number | null>(null);

  useEffect(() => {
    const t = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const list = await getClients();
        setClients(list);
        setClientsError(null);
      } catch (e) {
        setClientsError(e instanceof Error ? e.message : 'Failed to load clients');
      }
    })();
  }, []);

  useEffect(() => {
    getPrivacyBudget()
      .then((b) => {
        setBudgetUsed(b.total_epsilon_used);
        setBudgetCap(b.budget_cap);
      })
      .catch(() => { setBudgetUsed(null); setBudgetCap(null); });
  }, []);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem('federatedQueryHistory');
      if (raw) {
        const parsed = JSON.parse(raw) as { question: string; createdAt: string }[];
        setRecentQuestions(parsed.slice(0, 5).map((h) => ({
          q: h.question,
          time: new Date(h.createdAt).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }),
          status: 'completed' as const,
        })));
      } else setRecentQuestions([]);
    } catch {
      setRecentQuestions([]);
    }
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

      <div className="flex flex-wrap justify-between items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-sm text-gray-500 mt-0.5">Federated RAG overview</p>
        </div>
        <span className="text-sm text-gray-500">
          Last updated: {new Date().toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main action */}
        <motion.div variants={item} className="lg:col-span-2">
          <Card className="h-full bg-gradient-to-br from-indigo-600 via-indigo-700 to-indigo-800 text-white border-0 shadow-lg overflow-hidden">
            <div className="relative z-10 p-6">
              <div className="flex items-center gap-2 mb-2">
                <Database className="w-6 h-6 text-indigo-200" />
                <h2 className="text-xl font-bold">Federated query</h2>
              </div>
              <p className="text-indigo-100 mb-6 max-w-lg text-sm leading-relaxed">
                Query medical literature across federated nodes. Only metadata and truncated abstracts leave clients; no raw patient data.
              </p>
              <Link to="/ask">
                <Button variant="secondary" size="lg" rightIcon={<ArrowRight className="w-4 h-4" />}>
                  Ask a question
                </Button>
              </Link>
            </div>
            <div className="absolute right-0 bottom-0 opacity-10 pointer-events-none">
              <Database className="w-48 h-48 translate-x-8 translate-y-8" />
            </div>
          </Card>
        </motion.div>

        {/* Privacy budget – real data */}
        <motion.div variants={item}>
          <Card className="h-full flex flex-col">
            <div className="flex items-center gap-2 mb-4">
              <Shield className="w-5 h-5 text-indigo-600" />
              <h3 className="text-lg font-semibold text-gray-900">Privacy budget (ε)</h3>
            </div>
            {isLoading ? (
              <Skeleton circle width={120} height={120} className="mx-auto" />
            ) : (
              <Gauge
                value={
                  budgetCap != null && budgetCap > 0
                    ? Math.max(0, 1 - (budgetUsed ?? 0) / budgetCap)
                    : budgetUsed != null
                      ? Math.max(0, 1 - budgetUsed)
                      : 0.7
                }
                label="Remaining"
              />
            )}
            <div className="mt-4 text-center space-y-1">
              <p className="text-sm text-gray-600">
                Used: <span className="font-mono font-medium">{budgetUsed != null ? budgetUsed.toFixed(2) : '—'}</span> ε
              </p>
              {budgetCap != null && (
                <p className="text-xs text-gray-500">Cap: {budgetCap} ε</p>
              )}
            </div>
          </Card>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Client status – real doc counts */}
        <motion.div variants={item} className="lg:col-span-1">
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Federated nodes</h3>
              {clients.length > 0 && (
                <span className="text-xs text-gray-500">
                  {clients.reduce((s, c) => s + c.docs, 0).toLocaleString()} docs total
                </span>
              )}
            </div>
            {clientsError && (
              <p className="text-xs text-red-500 mb-2">{clientsError}</p>
            )}
            <div className="space-y-4">
              {(clients.length ? clients : [
                { id: 0, name: 'Client 0', status: 'online', docs: 0, version: 'v2.1.0' },
              ]).map((client) => (
                <div
                  key={client.id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center">
                    <div
                      className={`w-2 h-2 rounded-full mr-3 ${client.status === 'online' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {client.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {client.docs.toLocaleString()} documents
                      </p>
                    </div>
                  </div>
                  <Badge variant={client.status === 'online' ? 'success' : 'neutral'}>
                    {client.status}
                  </Badge>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* Recent questions – from history */}
        <motion.div variants={item} className="lg:col-span-1">
          <Card>
            <div className="flex items-center gap-2 mb-4">
              <MessageSquare className="w-5 h-5 text-indigo-600" />
              <h3 className="text-lg font-semibold text-gray-900">Recent questions</h3>
            </div>
            <div className="space-y-4">
              {(recentQuestions.length ? recentQuestions : [
                {
                  q: 'Ask your first question to see history here.',
                  time: '',
                  status: 'completed' as const,
                },
              ]).map((item, idx) =>
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

        {/* Activity sample */}
        <motion.div variants={item} className="lg:col-span-1">
          <Card>
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-5 h-5 text-indigo-600" />
              <h3 className="text-lg font-semibold text-gray-900">Query activity</h3>
            </div>
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={activitySample}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                    formatter={(v: number) => [`${v} queries`, 'Queries']}
                  />
                  <Line
                    type="monotone"
                    dataKey="queries"
                    stroke="#4F46E5"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-gray-500 mt-2">Sample distribution (demo)</p>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  );
}