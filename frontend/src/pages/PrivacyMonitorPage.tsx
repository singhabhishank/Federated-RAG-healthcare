import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Gauge } from '../components/ui/Gauge';
import { Table } from '../components/ui/Table';
import { Badge } from '../components/ui/Badge';
import { Shield, CheckCircle } from 'lucide-react';
import { getPrivacyBudget, type PrivacyBudgetResponse } from '../api';

const AUDIT_LOGS_PLACEHOLDER = [
  { id: 1, action: 'Federated Query', user: '—', time: '—', status: 'Approved', budget: '— ε' },
  { id: 2, action: '—', user: '—', time: '—', status: 'Approved', budget: '0 ε' },
];

export function PrivacyMonitorPage() {
  const [budget, setBudget] = useState<PrivacyBudgetResponse | null>(null);
  const [budgetError, setBudgetError] = useState<string | null>(null);

  useEffect(() => {
    getPrivacyBudget()
      .then(setBudget)
      .catch(() => setBudgetError('Could not load privacy budget'));
  }, []);

  const auditLogs = AUDIT_LOGS_PLACEHOLDER;
  const columns = [
    { header: 'Action', accessor: 'action' },
    { header: 'User', accessor: 'user' },
    { header: 'Time', accessor: 'time' },
    {
      header: 'Status',
      accessor: (item: { status: string }) => (
        <Badge variant={item.status === 'Approved' ? 'success' : 'warning'}>{item.status}</Badge>
      ),
    },
    { header: 'Budget Cost', accessor: 'budget', className: 'font-mono text-xs' },
  ];

  const cap = budget?.budget_cap ?? 1;
  const used = budget?.total_epsilon_used ?? 0;
  const remaining = budget?.remaining ?? (cap - used);
  const gaugeValue = cap > 0 ? Math.max(0, Math.min(1, remaining / cap)) : 0;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Privacy Monitor</h1>

      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="flex flex-col items-center justify-center p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Privacy Budget (ε)
          </h3>
          {budgetError ? (
            <p className="text-sm text-amber-600">Budget unavailable</p>
          ) : (
            <>
              <Gauge value={gaugeValue} size={160} />
              <div className="text-center mt-2">
                <p className="text-sm text-gray-500">
                  Used: {used.toFixed(2)} ε {budget?.budget_cap != null ? ` · Cap: ${cap} ε` : ''}
                </p>
                {budget?.budget_cap != null && (
                  <p className="text-sm text-gray-500">Remaining: {remaining.toFixed(2)} ε</p>
                )}
                <p className="text-xs text-gray-400 mt-1">
                  Tracked per query; formal composition bounds depend on composition method.
                </p>
              </div>
            </>
          )}
        </Card>

        <Card className="flex flex-col justify-center p-6 space-y-6">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-medium text-gray-900">
                Delta Parameter (δ)
              </h3>
              <span className="text-2xl font-bold text-indigo-600 font-mono">
                {budget?.delta != null ? String(budget.delta) : '1e-5'}
              </span>
            </div>
            <p className="text-sm text-gray-500">
              Probability of privacy breach is mathematically bounded.
            </p>
          </div>
          <div className="border-t border-gray-100 pt-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-medium text-gray-900">
                Noise Mechanism
              </h3>
              <Badge variant="primary">Gaussian</Badge>
            </div>
            <p className="text-sm text-gray-500">
              Applied to all vector embeddings before aggregation.
            </p>
          </div>
        </Card>

        <Card className="p-0 overflow-hidden bg-indigo-900 text-white flex flex-col justify-center items-center text-center">
          <div className="p-8">
            <Shield className="w-16 h-16 text-teal-400 mx-auto mb-4" />
            <h3 className="text-xl font-bold mb-2">Metadata-Only Mode</h3>
            <p className="text-indigo-200 mb-6">
              Raw text transfer is strictly disabled at the protocol level.
            </p>
            <div className="inline-flex items-center bg-teal-500/20 text-teal-300 px-3 py-1 rounded-full text-sm font-medium border border-teal-500/30">
              <CheckCircle className="w-4 h-4 mr-2" />
              Active & Enforced
            </div>
          </div>
        </Card>
      </div>

      {/* Audit Log */}
      <Card title="Privacy Audit Log" className="overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center">
          <h3 className="text-lg font-bold text-gray-900">Audit Log</h3>
          <Button variant="secondary" size="sm">
            Export Log
          </Button>
        </div>
        <Table data={auditLogs} columns={columns} />
      </Card>
    </div>);

}