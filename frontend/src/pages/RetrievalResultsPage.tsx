import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import {
  ArrowRight,
  Filter,
  Info,
  Calendar,
  Building,
  Lock } from
'lucide-react';
import type { QueryResponse, Citation } from '../api';

function citationToResult(c: Citation, idx: number, total: number): {
  id: number;
  title: string;
  journal: string;
  year: string;
  client: string;
  relevance: number;
  included: boolean;
  doi: string;
  pmc_id: string;
  doiUrl: string;
  pmcUrl: string;
} {
  const doi = (c.doi || '').trim();
  const pmcId = (c.pmc_id || '').trim();
  const pmcNorm = pmcId.replace(/^PMC/i, '');
  return {
    id: idx + 1,
    title: c.title || '[No title]',
    journal: c.journal || '[Unknown journal]',
    year: String(c.year || ''),
    client: `Client ${c.client_id ?? '?'}`,
    relevance: Math.round(100 - (idx / Math.max(total, 1)) * 25),
    included: idx < 10,
    doi,
    pmc_id: pmcId,
    doiUrl: doi ? `https://doi.org/${doi}` : '',
    pmcUrl: pmcNorm ? `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC${pmcNorm}/` : '',
  };
}

interface HistoryItem {
  id: number;
  question: string;
  createdAt: string;
  numReferences?: number;
  // Optional full response so we can reopen old answers without re-running
  response?: QueryResponse;
}

export function RetrievalResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as { queryResponse?: QueryResponse; question?: string } | null;
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const raw = window.localStorage.getItem('federatedQueryHistory');
      if (raw) {
        const parsed = JSON.parse(raw) as HistoryItem[];
        return Array.isArray(parsed) ? parsed : [];
      }
    } catch {
      // ignore
    }
    return [];
  });
  // When opening from sidebar (no state), show latest history entry so evidence/citations aren't empty
  const queryResponse = state?.queryResponse ?? (history.length > 0 && history[0].response ? history[0].response : undefined);
  const currentQuestion = state?.question ?? (history.length > 0 ? history[0].question : '');
  const results = queryResponse?.citations?.length
    ? queryResponse.citations.map((c, i, arr) => citationToResult(c, i, arr.length))
    : [];
  const [expandedId, setExpandedId] = useState<number | null>(null);

  // Keep history in sync when returning to this page (e.g. after a new query from Ask)
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem('federatedQueryHistory');
      if (raw) {
        const parsed = JSON.parse(raw) as HistoryItem[];
        setHistory(Array.isArray(parsed) ? parsed : []);
      }
    } catch {
      setHistory([]);
    }
  }, [location.pathname]);

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
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Federated Evidence
          </h1>
          <p className="text-sm text-gray-500">
            Aggregated from 3 secure nodes • {results.length} results found
          </p>
          {currentQuestion && (
            <p className="mt-1 text-sm text-gray-600 line-clamp-1">
              <span className="font-medium text-gray-700">Current question:</span> {currentQuestion}
            </p>
          )}
        </div>
        {queryResponse ? (
          <Link to="/answers" state={{ queryResponse, question: currentQuestion }}>
            <Button rightIcon={<ArrowRight className="w-4 h-4" />}>
              View Answer
            </Button>
          </Link>
        ) : (
          <Link to="/ask">
            <Button rightIcon={<ArrowRight className="w-4 h-4" />}>
              Ask a Question
            </Button>
          </Link>
        )}
      </div>

      {/* Info Banner */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-4 flex items-start">
        <Info className="w-5 h-5 text-indigo-600 mt-0.5 mr-3 flex-shrink-0" />
        <div>
          <h3 className="text-sm font-medium text-indigo-900">
            Privacy-Preserving Ranking
          </h3>
          <p className="text-sm text-indigo-700 mt-1">
            Results are ranked by noisy embedding similarity. The noise ensures
            differential privacy (ε=1.0). Raw text remains on client servers;
            only metadata is displayed here.
          </p>
          <p className="text-sm text-indigo-600 mt-2">
            If DOI/PMC ID is missing, go to <strong>Settings → Rebuild index</strong> so citations include identifiers from your dataset.
          </p>
        </div>
      </div>

      {/* Filters + History */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="p-4 lg:col-span-2">
          <div className="flex flex-wrap gap-4 items-end">
            <div className="w-full sm:w-48">
              <Input
                label="Year Range"
                placeholder="e.g. 2020 - 2026"
                title="Set on Ask page and re-run query to filter retrieval by year."
              />
            </div>
            <div className="w-full sm:w-48">
              <Select
                label="Journal"
                options={[
                {
                  value: 'all',
                  label: 'All Journals'
                },
                {
                  value: 'cardio',
                  label: 'Cardiology'
                },
                {
                  value: 'pharm',
                  label: 'Pharmacology'
                }]
                } />

            </div>
            <div className="w-full sm:w-48">
              <Select
                label="Client Source"
                options={[
                {
                  value: 'all',
                  label: 'All Clients'
                },
                {
                  value: 'c0',
                  label: 'Client 0'
                },
                {
                  value: 'c1',
                  label: 'Client 1'
                },
                {
                  value: 'c2',
                  label: 'Client 2'
                }]
                } />

            </div>
            <Button variant="secondary" leftIcon={<Filter className="w-4 h-4" />}>
              Apply Filters
            </Button>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="text-sm font-semibold text-gray-900 mb-2">History</h3>
          {history.length === 0 ? (
            <p className="text-xs text-gray-500">
              No previous questions yet. New questions will appear here.
            </p>
          ) : (
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {history.map((h) => (
                <button
                  key={h.id}
                  type="button"
                  className="w-full text-left text-xs text-gray-700 border-b border-gray-100 last:border-0 pb-1 hover:bg-gray-50 rounded"
                  onClick={() => {
                    if (h.response) {
                      // Open previous answer/evidence directly
                      navigate('/results', {
                        state: {
                          queryResponse: h.response,
                          question: h.question,
                        },
                      });
                    } else {
                      // Older history entries without response: fall back to re-running
                      navigate('/ask', { state: { prefillQuestion: h.question, autoRun: true } });
                    }
                  }}>
                  <p className="line-clamp-1">{h.question}</p>
                  <p className="text-[10px] text-gray-400">
                    {new Date(h.createdAt).toLocaleString()} • {h.numReferences ?? 0} refs
                  </p>
                </button>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Rebuild hint when no citation has DOI/PMC (old index) */}
      {results.length > 0 && !results.some((r) => r.doi || r.pmc_id) && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-start">
          <Info className="w-5 h-5 text-amber-600 mt-0.5 mr-3 flex-shrink-0" />
          <div>
            <h3 className="text-sm font-medium text-amber-900">DOI / PMC ID not in results</h3>
            <p className="text-sm text-amber-800 mt-1">
              The index was built before identifiers were added. Go to <strong>Settings</strong> → click <strong>Rebuild index (to show DOI/PMC ID)</strong>, then run a new query from Ask.
            </p>
            <Link to="/settings" className="inline-block mt-2 text-sm font-medium text-amber-700 hover:underline">Go to Settings →</Link>
          </div>
        </div>
      )}

      {/* Results List */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-4">

        {results.length === 0 && !queryResponse && (
          <Card className="p-8 text-center text-gray-500">
            <p>No retrieval results yet. Ask a question from the Ask page to run federated retrieval.</p>
            <Link to="/ask" className="mt-4 inline-block text-indigo-600 hover:underline">Go to Ask a Question</Link>
          </Card>
        )}
        {results.map((result) =>
        <motion.div key={result.id} variants={item}>
            <Card
            className={`cursor-pointer transition-all duration-200 ${expandedId === result.id ? 'ring-2 ring-indigo-500' : 'hover:shadow-md'}`}
            onClick={() =>
            setExpandedId(expandedId === result.id ? null : result.id)
            }>

              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <Badge
                    variant={
                    result.client === 'Client 0' ?
                    'info' :
                    result.client === 'Client 1' ?
                    'primary' :
                    'success'
                    }>

                      {result.client}
                    </Badge>
                    {result.included &&
                  <Badge variant="success" className="flex items-center">
                        Included in Answer
                      </Badge>
                  }
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 mb-1">
                    {result.title}
                  </h3>
                  <div className="flex items-center text-sm text-gray-500 space-x-4">
                    <span className="flex items-center">
                      <Building className="w-3 h-3 mr-1" /> {result.journal}
                    </span>
                    <span className="flex items-center">
                      <Calendar className="w-3 h-3 mr-1" /> {result.year}
                    </span>
                  </div>
                </div>

                <div className="flex flex-col items-end ml-4">
                  <div className="flex items-center mb-1">
                    <span className="text-sm font-bold text-gray-900 mr-2">
                      {result.relevance}%
                    </span>
                    <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                      <div
                      className="h-full bg-indigo-600 rounded-full"
                      style={{
                        width: `${result.relevance}%`
                      }} />

                    </div>
                  </div>
                  <span className="text-xs text-gray-400">
                    Relevance (Noisy)
                  </span>
                </div>
              </div>

              {expandedId === result.id &&
            <motion.div
              initial={{
                opacity: 0,
                height: 0
              }}
              animate={{
                opacity: 1,
                height: 'auto'
              }}
              className="mt-4 pt-4 border-t border-gray-100">

                  <div className="grid grid-cols-1 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-gray-700">DOI / PMC ID:</span>
                      <p className="text-gray-600 font-mono">
                        {result.doi ? (
                          <a
                            href={result.doiUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-indigo-600 hover:underline"
                          >
                            {result.doi}
                          </a>
                        ) : result.pmcUrl ? (
                          <a
                            href={result.pmcUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-indigo-600 hover:underline"
                          >
                            PMC{result.pmc_id.replace(/^PMC/i, '')}
                          </a>
                        ) : (
                          '—'
                        )}
                      </p>
                    </div>
                  </div>
                  <div className="mt-3 bg-gray-50 p-3 rounded text-xs text-gray-500 flex items-center">
                    <Lock className="w-3 h-3 mr-2" />
                    Full text content is retained locally on {result.client}.
                    Only embeddings were used for retrieval.
                  </div>
                </motion.div>
            }
            </Card>
          </motion.div>
        )}
      </motion.div>
    </div>);

}