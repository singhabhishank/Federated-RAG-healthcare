import { useState, useCallback, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import {
  Copy,
  FileText,
  Share2,
  Check,
  AlertTriangle,
} from 'lucide-react';
import type { QueryResponse, Citation } from '../api';

function formatCitationsForText(citations: Citation[]): string {
  return citations
    .map((c, i) => {
      const ref = `${i + 1}. ${c.title || '[No title]'} (${c.journal || ''}, ${c.year || ''}) [Client ${c.client_id ?? '?'}]`;
      const idPart = c.doi ? ` DOI: ${c.doi}` : c.pmc_id ? ` PMC ID: ${c.pmc_id}` : '';
      return ref + (idPart ? idPart : '');
    })
    .join('\n');
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Convert markdown-style **bold** and *bold* to HTML <strong>; strip asterisks for display. */
function answerToHtml(text: string): string {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<strong>$1</strong>')
    .replace(/\_(.+?)\_/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br />');
}

/** Strip ** and * for plain-text share/copy (keep words, no markdown). */
function stripMarkdownBold(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/\_(.+?)\_/g, '$1');
}

function buildFullContent(question: string, answer: string, citations: Citation[]): string {
  const lines = [];
  if (question.trim()) lines.push(`Question: ${question.trim()}\n`);
  lines.push(stripMarkdownBold(answer.trim()));
  if (citations.length > 0) {
    lines.push('\n\nSources:\n' + formatCitationsForText(citations));
  }
  return lines.join('');
}

interface HistoryEntry {
  id: number;
  question: string;
  createdAt: string;
  numReferences?: number;
  response?: QueryResponse;
}

function loadHistory(): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem('federatedQueryHistory');
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryEntry[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function AnswerOutputPage() {
  const location = useLocation();
  const state = location.state as { queryResponse?: QueryResponse; question?: string } | null;
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  // Real data: from navigation state or latest history so Evidence is never dummy when user has run a query
  const queryResponse = state?.queryResponse ?? (history.length > 0 && history[0].response ? history[0].response : undefined);
  const question = state?.question ?? (history.length > 0 ? history[0].question : '');
  const fromHistory = !state?.queryResponse && !!queryResponse;
  const [copied, setCopied] = useState(false);
  const [shareStatus, setShareStatus] = useState<'idle' | 'shared' | 'copied' | 'error'>('idle');

  // Keep in sync with latest query (real-time: re-read when visiting this page)
  useEffect(() => {
    setHistory(loadHistory());
  }, [location.pathname]);

  const citations: Citation[] = queryResponse?.citations ?? [];
  const answer = queryResponse?.answer ?? '';
  const abstractsIncluded = queryResponse?.abstracts_included ?? false;
  const fullContent = buildFullContent(question, answer, citations);

  const handleCopy = useCallback(async () => {
    if (!fullContent.trim()) return;
    try {
      await navigator.clipboard.writeText(fullContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      setCopied(false);
    }
  }, [fullContent]);

  const handleShare = useCallback(async () => {
    if (!answer.trim()) return;
    const title = 'Federated RAG Answer';
    const rawSnippet = question.trim()
      ? `${question}\n\n${answer.slice(0, 500)}${answer.length > 500 ? '…' : ''}`
      : answer.slice(0, 1000);
    const text = stripMarkdownBold(rawSnippet);
    const url = window.location.href;
    try {
      if (typeof navigator.share === 'function' && navigator.canShare?.({ title, text, url }) !== false) {
        await navigator.share({ title, text, url });
        setShareStatus('shared');
      } else {
        await navigator.clipboard.writeText(`${title}\n\n${text}\n\n${url}`);
        setShareStatus('copied');
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      try {
        await navigator.clipboard.writeText(`${title}\n\n${text}\n\n${url}`);
        setShareStatus('copied');
      } catch {
        setShareStatus('error');
      }
    }
    setTimeout(() => setShareStatus('idle'), 2500);
  }, [question, answer]);

  const handlePDF = useCallback(() => {
    if (!answer.trim()) return;
    const citationList = citations
      .map((c, i) => {
        const idPart = c.doi
          ? ` <a href="https://doi.org/${escapeHtml(c.doi)}" target="_blank" rel="noopener">${escapeHtml(c.doi)}</a>`
          : c.pmc_id
            ? ` PMC ID: <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC${escapeHtml(String(c.pmc_id).replace(/^PMC/i, ''))}/" target="_blank" rel="noopener">${escapeHtml(c.pmc_id)}</a>`
            : '';
        return `<li><strong>${i + 1}.</strong> ${escapeHtml(c.title || '[No title]')} — ${escapeHtml(c.journal || '')}, ${c.year || ''} (Client ${c.client_id ?? '?'})${idPart ? ' • ' + idPart : ''}</li>`;
      })
      .join('');
    const answerHtml = answerToHtml(answer);
    const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Federated RAG Answer</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 24px auto; padding: 0 16px; color: #1f2937; line-height: 1.6; }
    h1 { font-size: 1.25rem; color: #374151; margin-bottom: 8px; }
    .answer { margin: 16px 0; }
    .answer strong { font-weight: 700; color: #1f2937; }
    .sources { margin-top: 24px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 0.875rem; }
    .sources ul { padding-left: 1.25rem; }
    .disclaimer { margin-top: 24px; padding: 12px; background: #fef3c7; border-left: 4px solid #f59e0b; font-size: 0.875rem; }
  </style>
</head>
<body>
  ${question.trim() ? `<h1>Question</h1><p>${escapeHtml(question)}</p>` : ''}
  <h1>Answer</h1>
  <div class="answer">${answerHtml}</div>
  ${citations.length > 0 ? `<div class="sources"><h2>Sources</h2><ul>${citationList}</ul></div>` : ''}
  <div class="disclaimer">This response was generated from federated retrieval with differential privacy. ${abstractsIncluded ? 'Grounded in document abstracts.' : 'Metadata-only (no document content).'} Not medical advice.</div>
</body>
</html>`;
    const printWindow = window.open('', '_blank');
    if (!printWindow) {
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.target = '_blank';
      a.rel = 'noopener';
      a.download = 'federated-rag-answer.html';
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      return;
    }
    printWindow.document.write(html);
    printWindow.document.close();
    const doPrint = () => {
      printWindow.print();
      printWindow.onafterprint = () => printWindow.close();
    };
    if (printWindow.document.readyState === 'complete') doPrint();
    else printWindow.onload = doPrint;
  }, [question, answer, citations, abstractsIncluded]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Main Answer Column */}
      <div className="lg:col-span-2 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-gray-900">Generated Answer</h1>
            {queryResponse && (
              <span
                className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                  abstractsIncluded
                    ? 'bg-emerald-100 text-emerald-800'
                    : 'bg-amber-100 text-amber-800'
                }`}
                title={abstractsIncluded ? 'Answer grounded in retrieved document abstracts' : 'Answer from metadata only (title, journal, year); not grounded in document content'}
              >
                {abstractsIncluded ? 'Grounded in content' : 'Metadata-only'}
              </span>
            )}
          </div>
          <div className="flex space-x-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleCopy}
              disabled={!answer.trim()}
              leftIcon={copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
            >
              {copied ? 'Copied' : 'Copy'}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<FileText className="w-4 h-4" />}
              onClick={handlePDF}
              disabled={!answer.trim()}
            >
              PDF
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<Share2 className="w-4 h-4" />}
              onClick={handleShare}
              disabled={!answer.trim()}
            >
              {shareStatus === 'shared' ? 'Shared' : shareStatus === 'copied' ? 'Link copied' : shareStatus === 'error' ? 'Failed' : 'Share'}
            </Button>
          </div>
        </div>

        <Card className="min-h-[600px] relative">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            {!queryResponse ? (
              <div className="text-gray-500 text-center py-12">
                <p>No answer to display. Run a query from the Ask page first.</p>
                <Link to="/ask" className="mt-4 inline-block text-indigo-600 hover:underline">Ask a Question</Link>
              </div>
            ) : (
              <>
              {fromHistory && (
                <p className="text-xs text-indigo-600 mb-2">Showing latest from your history</p>
              )}
              <div className="prose prose-indigo max-w-none">
                <div
                  className="text-gray-800 leading-relaxed [&_strong]:font-semibold [&_strong]:text-gray-900"
                  style={{ whiteSpace: 'pre-wrap' }}
                  dangerouslySetInnerHTML={{ __html: answerToHtml(answer) }}
                />
                <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded-r-lg mt-8">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <AlertTriangle className="h-5 w-5 text-amber-400" aria-hidden="true" />
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-amber-800">Limitations & Safety</h3>
                      <div className="mt-2 text-sm text-amber-700">
                        <p>
                          {abstractsIncluded
                            ? 'This response is grounded in retrieved document abstracts (truncated) and metadata, with differential privacy. Not medical advice—consult qualified professionals for clinical decisions.'
                            : 'This response is based on metadata only (title, journal, year); the model did not read document content, so citations are not evidence of specific claims. Not medical advice—consult qualified professionals for clinical decisions.'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              </>
            )}
          </motion.div>
        </Card>
      </div>

      {/* Evidence Sources - real citations from last query or history */}
      <div className="lg:col-span-1">
        <h2 className="text-lg font-bold text-gray-900 mb-4">Evidence Sources</h2>
        {queryResponse && citations.length > 0 && (
          <p className="text-xs text-gray-500 mb-2">{citations.length} source{citations.length !== 1 ? 's' : ''} from federated retrieval</p>
        )}
        <div className="space-y-4">
          {citations.length === 0 && queryResponse && (
            <p className="text-sm text-gray-500">No citations returned for this query.</p>
          )}
          {citations.map((citation, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <Card className="hover:shadow-md transition-shadow cursor-pointer" padding="sm">
                <div className="flex items-start">
                  <span className="flex-shrink-0 flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-800 text-xs font-bold mr-3">
                    {idx + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <h4 className="text-sm font-medium text-gray-900 line-clamp-2 mb-1">
                      {citation.title || '[No title]'}
                    </h4>
                    <div className="flex flex-wrap gap-2 text-xs text-gray-500 mb-1">
                      <span>{citation.journal || ''}</span>
                      <span>•</span>
                      <span>{citation.year || ''}</span>
                      {(citation.doi || citation.pmc_id) && (
                        <>
                          <span>•</span>
                          {citation.doi ? (
                            <a
                              href={`https://doi.org/${citation.doi}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-indigo-600 hover:underline font-mono"
                            >
                              {citation.doi}
                            </a>
                          ) : (
                            <a
                              href={`https://www.ncbi.nlm.nih.gov/pmc/articles/PMC${String(citation.pmc_id).replace(/^PMC/i, '')}/`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-indigo-600 hover:underline font-mono"
                            >
                              PMC{citation.pmc_id}
                            </a>
                          )}
                        </>
                      )}
                    </div>
                    {citation.abstract && (
                      <p className="text-xs text-gray-600 line-clamp-3 mt-1">{citation.abstract}</p>
                    )}
                    <Badge variant="neutral" size="sm" className="mt-2">
                      Client {citation.client_id ?? '?'}
                    </Badge>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );

}