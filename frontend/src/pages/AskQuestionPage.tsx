import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Textarea } from '../components/ui/Textarea';
import { Slider } from '../components/ui/Slider';
import { Toggle } from '../components/ui/Toggle';
import { Select } from '../components/ui/Select';
import { Stepper, Step } from '../components/ui/Stepper';
import { Tooltip } from '../components/ui/Tooltip';
import {
  Lock,
  Zap,
  Database,
  Shield,
  GitMerge,
  MessageSquare,
  ChevronDown,
  ChevronUp } from
'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { query, type QueryResponse } from '../api';

export function AskQuestionPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);
  const [yearMin, setYearMin] = useState<number | ''>('');
  const [yearMax, setYearMax] = useState<number | ''>('');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [steps, setSteps] = useState<Step[]>([
  {
    id: '1',
    title: 'Query Embedded',
    status: 'pending',
    icon: <Zap className="w-4 h-4" />
  },
  {
    id: '2',
    title: 'Local Retrieval',
    status: 'pending',
    description: 'Scanning 3 nodes...',
    icon: <Database className="w-4 h-4" />
  },
  {
    id: '3',
    title: 'DP Noise Applied',
    status: 'pending',
    icon: <Shield className="w-4 h-4" />
  },
  {
    id: '4',
    title: 'Aggregation',
    status: 'pending',
    icon: <GitMerge className="w-4 h-4" />
  },
  {
    id: '5',
    title: 'Answer Generation',
    status: 'pending',
    icon: <MessageSquare className="w-4 h-4" />
  }]
  );

  // Prefill question (and optionally auto-run) when coming from history
  useEffect(() => {
    const state = location.state as { prefillQuestion?: string; autoRun?: boolean } | null;
    if (state?.prefillQuestion) {
      setQuestion(state.prefillQuestion);
      if (state.autoRun && !isRunning) {
        // Clear the state so we don't auto-run again on back/forward
        navigate(location.pathname, { replace: true, state: {} });
        // Defer run to next tick so state updates apply
        setTimeout(() => {
          void handleRun();
        }, 0);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);

  const setStep = (idx: number, status: Step['status']) => {
    setSteps((prev) =>
      prev.map((s, i) => ({
        ...s,
        status: i === idx ? status : i < idx ? 'completed' : s.status,
      }))
    );
  };

  const handleRun = async () => {
    const q = question.trim();
    if (!q) {
      setError('Please enter a question.');
      return;
    }
    setError(null);
    setIsRunning(true);
    setSteps((prev) => prev.map((s) => ({ ...s, status: 'pending' as const })));

    const stepDuration = 800;
    try {
      setStep(0, 'current');
      await new Promise((r) => setTimeout(r, stepDuration));
      setStep(0, 'completed');
      setStep(1, 'current');
      await new Promise((r) => setTimeout(r, stepDuration));
      setStep(1, 'completed');
      setStep(2, 'current');
      await new Promise((r) => setTimeout(r, stepDuration));
      setStep(2, 'completed');
      setStep(3, 'current');
      await new Promise((r) => setTimeout(r, stepDuration));
      setStep(3, 'completed');
      setStep(4, 'current');

      const result: QueryResponse = await query({
        question: q,
        top_k_per_client: topK,
        year_min: typeof yearMin === 'number' ? yearMin : undefined,
        year_max: typeof yearMax === 'number' ? yearMax : undefined,
      });

      // Persist to local history (per-browser), including full response
      try {
        const key = 'federatedQueryHistory';
        const existingRaw = window.localStorage.getItem(key);
        const existing = existingRaw ? JSON.parse(existingRaw) as any[] : [];
        const entry = {
          id: Date.now(),
          question: q,
          createdAt: new Date().toISOString(),
          numReferences: result.num_references,
          response: result,
        };
        const next = [entry, ...existing].slice(0, 50);
        window.localStorage.setItem(key, JSON.stringify(next));
      } catch {
        // Ignore history errors
      }

      setStep(4, 'completed');
      navigate('/results', { state: { queryResponse: result, question: q } });
    } catch (e) {
      setStep(4, 'error');
      setError(e instanceof Error ? e.message : 'Query failed.');
    } finally {
      setIsRunning(false);
    }
  };
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-[calc(100vh-8rem)]">
      {/* Left Column - Input & Settings */}
      <div className="lg:col-span-7 flex flex-col h-full">
        <Card className="flex-1 flex flex-col h-full">
          <h1 className="text-2xl font-bold text-gray-900 mb-6">
            Ask a Question
          </h1>

          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
              {error}
            </div>
          )}
          <div className="flex-1 mb-6">
            <Textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., What are the reported cardiovascular side effects of long-term ibuprofen use in elderly patients?"
              className="h-full text-lg p-6 resize-none"
              disabled={isRunning} />

          </div>

          <div className="border-t border-gray-100 pt-4">
            <button
              onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              className="flex items-center text-sm font-medium text-gray-600 hover:text-indigo-600 mb-4 transition-colors">

              {isSettingsOpen ?
              <ChevronDown className="w-4 h-4 mr-1" /> :

              <ChevronUp className="w-4 h-4 mr-1" />
              }
              Advanced Configuration
            </button>

            <motion.div
              initial={false}
              animate={{
                height: isSettingsOpen ? 'auto' : 0,
                opacity: isSettingsOpen ? 1 : 0
              }}
              className="overflow-hidden">

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6 bg-gray-50 p-4 rounded-lg">
                <div>
                  <Slider
                    value={[topK]}
                    onValueChange={(v) => setTopK(v[0] ?? 5)}
                    max={20}
                    min={1}
                    label="Top-K per Client" />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 block mb-1">Year range (filter retrieval)</label>
                  <div className="flex gap-2 items-center">
                    <input
                      type="number"
                      placeholder="Min"
                      className="w-20 rounded border border-gray-300 px-2 py-1.5 text-sm"
                      value={yearMin === '' ? '' : yearMin}
                      onChange={(e) => setYearMin(e.target.value === '' ? '' : parseInt(e.target.value, 10))}
                      min={1900}
                      max={2100}
                    />
                    <span className="text-gray-500">–</span>
                    <input
                      type="number"
                      placeholder="Max"
                      className="w-20 rounded border border-gray-300 px-2 py-1.5 text-sm"
                      value={yearMax === '' ? '' : yearMax}
                      onChange={(e) => setYearMax(e.target.value === '' ? '' : parseInt(e.target.value, 10))}
                      min={1900}
                      max={2100}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex items-center mb-2">
                    <label className="text-sm font-medium text-gray-700 mr-2">
                      DP Noise Level (σ)
                    </label>
                    <Tooltip content="Higher noise = more privacy but less accuracy">
                      <Shield className="w-3 h-3 text-gray-400" />
                    </Tooltip>
                  </div>
                  <Slider
                    value={[0.1]}
                    onValueChange={() => {}}
                    max={1}
                    min={0.01}
                    step={0.01} />

                </div>
                <div>
                  <Select
                    label="LLM Provider"
                    options={[
                    {
                      value: 'openrouter',
                      label: 'OpenRouter (Default)'
                    },
                    {
                      value: 'groq',
                      label: 'Groq (Fastest)'
                    },
                    {
                      value: 'openai',
                      label: 'OpenAI GPT-4'
                    },
                    {
                      value: 'qwen',
                      label: 'Qwen 2.5 (Local)'
                    }]
                    } />

                </div>
                <div className="flex items-center justify-between">
                  <Toggle
                    checked={true}
                    onChange={() => {}}
                    label="Re-ranking" />

                  <div className="flex items-center text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded border border-amber-100">
                    <Lock className="w-3 h-3 mr-1" />
                    Metadata-only Mode
                  </div>
                </div>
              </div>
            </motion.div>

            <Button
              size="lg"
              className="w-full h-14 text-lg"
              onClick={handleRun}
              isLoading={isRunning}
              disabled={isRunning}>

              {isRunning ?
              'Running Federated Retrieval...' :
              'Run Federated Retrieval'}
            </Button>
          </div>
        </Card>
      </div>

      {/* Right Column - Progress Timeline */}
      <div className="lg:col-span-5">
        <Card className="h-full bg-gray-50/50">
          <h2 className="text-lg font-bold text-gray-900 mb-6">
            Execution Timeline
          </h2>
          <div className="pl-2">
            <Stepper steps={steps} orientation="vertical" />
          </div>

          {isRunning &&
          <motion.div
            initial={{
              opacity: 0
            }}
            animate={{
              opacity: 1
            }}
            className="mt-8 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">

              <h3 className="text-sm font-medium text-gray-900 mb-2">
                Live Logs
              </h3>
              <div className="font-mono text-xs text-gray-500 space-y-1">
                <p>{'>'} Initializing secure channel...</p>
                <p>{'>'} Broadcasting query vector to 3 nodes...</p>
                {steps[1].status === 'current' &&
              <p className="text-indigo-600">
                    {'>'} Waiting for Client 0, 1, 2...
                  </p>
              }
                {steps[2].status === 'current' &&
              <p className="text-indigo-600">
                    {'>'} Applying Gaussian mechanism (ε=1.0)...
                  </p>
              }
                {steps[3].status === 'current' &&
              <p className="text-indigo-600">
                    {'>'} Aggregating 15 encrypted results...
                  </p>
              }
              </div>
            </motion.div>
          }
        </Card>
      </div>
    </div>);

}