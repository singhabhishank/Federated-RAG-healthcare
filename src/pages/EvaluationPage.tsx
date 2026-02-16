import React from 'react';
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
  Radar } from
'recharts';
import { Play } from 'lucide-react';
const radarData = [
{
  subject: 'Relevance',
  A: 120,
  fullMark: 150
},
{
  subject: 'Accuracy',
  A: 98,
  fullMark: 150
},
{
  subject: 'Privacy',
  A: 145,
  fullMark: 150
},
{
  subject: 'Speed',
  A: 110,
  fullMark: 150
},
{
  subject: 'Safety',
  A: 130,
  fullMark: 150
},
{
  subject: 'Completeness',
  A: 85,
  fullMark: 150
}];

const barData = [
{
  name: 'Run 1',
  score: 85
},
{
  name: 'Run 2',
  score: 88
},
{
  name: 'Run 3',
  score: 92
},
{
  name: 'Run 4',
  score: 90
},
{
  name: 'Run 5',
  score: 95
}];

export function EvaluationPage() {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">System Evaluation</h1>
        <Button leftIcon={<Play className="w-4 h-4" />}>
          Run New Evaluation
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Performance Metrics">
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis />
                <Radar
                  name="Current System"
                  dataKey="A"
                  stroke="#4F46E5"
                  fill="#4F46E5"
                  fillOpacity={0.6} />

              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Retrieval Accuracy History">
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="score" fill="#0D9488" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
        {
          label: 'Retrieval Relevance',
          value: '92%',
          change: '+2.4%'
        },
        {
          label: 'Answer Quality',
          value: '4.8/5',
          change: '+0.1'
        },
        {
          label: 'Privacy Compliance',
          value: '100%',
          change: '0%'
        },
        {
          label: 'Avg Latency',
          value: '142ms',
          change: '-12ms'
        }].
        map((metric, idx) =>
        <Card key={idx} padding="sm">
            <p className="text-sm text-gray-500 mb-1">{metric.label}</p>
            <div className="flex items-end justify-between">
              <span className="text-2xl font-bold text-gray-900">
                {metric.value}
              </span>
              <span
              className={`text-xs font-medium ${metric.change.startsWith('+') || metric.change.startsWith('-') ? 'text-green-600' : 'text-gray-500'}`}>

                {metric.change}
              </span>
            </div>
          </Card>
        )}
      </div>
    </div>);

}