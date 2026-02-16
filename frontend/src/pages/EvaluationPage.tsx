import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { Play, Loader2 } from 'lucide-react';
import { getEvaluation, runEvaluate, type EvaluationResponse } from '../api';

function radarFromStats(s: EvaluationResponse['statistics']) {
  const rel = s?.avg_retrieval_relevance != null ? Math.round(s.avg_retrieval_relevance * 100) : 0;
  const qual = s?.avg_answer_quality != null ? Math.round(s.avg_answer_quality * 100) : 0;
  const priv = s?.privacy_compliance_rate != null ? Math.round(s.privacy_compliance_rate * 100) : 0;
  return [
    { subject: 'Relevance', A: rel, fullMark: 100 },
    { subject: 'Accuracy', A: qual, fullMark: 100 },
    { subject: 'Privacy', A: priv, fullMark: 100 },
    { subject: 'Completeness', A: qual, fullMark: 100 },
  ];
}

export function EvaluationPage() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setError(null);
    try {
      const res = await getEvaluation();
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load evaluation');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await runEvaluate(10);
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Evaluation failed. Is the backend running and is RAG initialized?');
    } finally {
      setRunning(false);
    }
  };

  const stats = data?.statistics ?? {};
  const hasNoRunYet = data?.message != null || (Object.keys(stats).length === 0 && (data?.results?.length ?? 0) === 0);
  const radarData = radarFromStats(stats);
  const barData = (data?.results ?? [])
    .slice(0, 10)
    .map((r: { retrieval_relevance_score?: number; question?: string }, i: number) => ({
      name: `Q${i + 1}`,
      score: r.retrieval_relevance_score != null ? Math.round(r.retrieval_relevance_score * 100) : 0,
    }));

  const metrics = [
    {
      label: 'Retrieval Relevance',
      value: stats.avg_retrieval_relevance != null ? `${Math.round(stats.avg_retrieval_relevance * 100)}%` : '—',
      change: '',
    },
    {
      label: 'Answer Quality',
      value: stats.avg_answer_quality != null ? (stats.avg_answer_quality * 5).toFixed(1) + '/5' : '—',
      change: '',
    },
    {
      label: 'Privacy Compliance',
      value: stats.privacy_compliance_rate != null ? `${Math.round(stats.privacy_compliance_rate * 100)}%` : '—',
      change: '',
    },
    {
      label: 'Successful Queries',
      value: stats.successful_queries != null ? `${stats.successful_queries}/${stats.total_questions ?? 0}` : '—',
      change: '',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">System Evaluation</h1>
        <Button
          leftIcon={running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          onClick={handleRun}
          disabled={running}
        >
          {running ? 'Running Evaluation...' : 'Run New Evaluation'}
        </Button>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">{error}</div>
      )}

      {loading ? (
        <Card className="p-8 text-center text-gray-500">Loading evaluation data...</Card>
      ) : hasNoRunYet ? (
        <Card className="p-8 text-center">
          <p className="text-gray-600 mb-2">No evaluation has been run yet.</p>
          <p className="text-sm text-gray-500 mb-6">
            Click &quot;Run New Evaluation&quot; to run the real evaluation on the backend (uses medical questions and writes results to the server).
          </p>
          <Button
            leftIcon={running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            onClick={handleRun}
            disabled={running}
          >
            {running ? 'Running Evaluation...' : 'Run New Evaluation'}
          </Button>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card title="Performance Metrics">
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis domain={[0, 100]} />
                    <Radar name="Current System" dataKey="A" stroke="#4F46E5" fill="#4F46E5" fillOpacity={0.6} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card title="Retrieval Relevance (per question)">
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData.length ? barData : [{ name: '—', score: 0 }]}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="score" fill="#0D9488" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {metrics.map((metric, idx) => (
              <Card key={idx} padding="sm">
                <p className="text-sm text-gray-500 mb-1">{metric.label}</p>
                <div className="flex items-end justify-between">
                  <span className="text-2xl font-bold text-gray-900">{metric.value}</span>
                  {metric.change && (
                    <span className="text-xs font-medium text-gray-500">{metric.change}</span>
                  )}
                </div>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  );

}