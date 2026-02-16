import React, { useState } from 'react';
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
  ExternalLink } from
'lucide-react';
export function AnswerOutputPage() {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Main Answer Column */}
      <div className="lg:col-span-2 space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">Generated Answer</h1>
          <div className="flex space-x-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleCopy}
              leftIcon={
              copied ?
              <Check className="w-4 h-4" /> :

              <Copy className="w-4 h-4" />

              }>

              {copied ? 'Copied' : 'Copy'}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<FileText className="w-4 h-4" />}>

              PDF
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<Share2 className="w-4 h-4" />}>

              Share
            </Button>
          </div>
        </div>

        <Card className="min-h-[600px] relative">
          <motion.div
            initial={{
              opacity: 0
            }}
            animate={{
              opacity: 1
            }}
            transition={{
              duration: 0.5
            }}>

            <div className="prose prose-indigo max-w-none">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Summary</h3>
              <p className="text-gray-800 leading-relaxed mb-6">
                Based on the aggregated evidence from 3 hospital nodes, ACE
                inhibitors remain a cornerstone therapy for heart failure
                management. Recent multi-center studies indicate a 15% reduction
                in mortality when compared to placebo{' '}
                <span className="inline-flex items-center justify-center px-1.5 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 cursor-pointer hover:bg-indigo-200">
                  [1]
                </span>
                . However, comparative analyses with ARBs suggest similar
                efficacy profiles with a lower incidence of cough-related side
                effects in the ARB group{' '}
                <span className="inline-flex items-center justify-center px-1.5 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 cursor-pointer hover:bg-indigo-200">
                  [3]
                </span>
                .
              </p>

              <h3 className="text-lg font-bold text-gray-900 mb-4">
                Clinical Considerations
              </h3>
              <ul className="list-disc pl-5 space-y-2 text-gray-800 mb-6">
                <li>
                  Monitor renal function and potassium levels within 1-2 weeks
                  of initiation.
                </li>
                <li>
                  Consider switching to ARBs if patient develops persistent dry
                  cough.
                </li>
                <li>
                  Beta-blocker co-administration is recommended for optimal
                  outcomes in elderly patients{' '}
                  <span className="inline-flex items-center justify-center px-1.5 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 cursor-pointer hover:bg-indigo-200">
                    [2]
                  </span>
                  .
                </li>
              </ul>

              <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded-r-lg mt-8">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <AlertTriangle
                      className="h-5 w-5 text-amber-400"
                      aria-hidden="true" />

                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-amber-800">
                      Limitations & Safety
                    </h3>
                    <div className="mt-2 text-sm text-amber-700">
                      <p>
                        This response is generated from metadata only using
                        differential privacy (ε=1.0). Not medical advice. For
                        clinical decisions, consult qualified professionals.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </Card>
      </div>

      {/* Citations Sidebar */}
      <div className="lg:col-span-1">
        <h2 className="text-lg font-bold text-gray-900 mb-4">
          Evidence Sources
        </h2>
        <div className="space-y-4">
          {[
          {
            id: 1,
            title: 'Efficacy of ACE Inhibitors in Heart Failure...',
            journal: 'Journal of Cardiology',
            year: 2024,
            client: 'Client 0'
          },
          {
            id: 2,
            title: 'Long-term Outcomes of Beta-Blocker Therapy...',
            journal: 'Geriatric Medicine',
            year: 2023,
            client: 'Client 1'
          },
          {
            id: 3,
            title: 'Comparative Analysis of ARBs vs ACE Inhibitors',
            journal: 'Heart Health Weekly',
            year: 2025,
            client: 'Client 2'
          }].
          map((citation, idx) =>
          <motion.div
            key={citation.id}
            initial={{
              opacity: 0,
              x: 20
            }}
            animate={{
              opacity: 1,
              x: 0
            }}
            transition={{
              delay: idx * 0.1
            }}>

              <Card
              className="hover:shadow-md transition-shadow cursor-pointer"
              padding="sm">

                <div className="flex items-start">
                  <span className="flex-shrink-0 flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-800 text-xs font-bold mr-3">
                    {citation.id}
                  </span>
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 line-clamp-2 mb-1">
                      {citation.title}
                    </h4>
                    <div className="flex flex-wrap gap-2 text-xs text-gray-500 mb-2">
                      <span>{citation.journal}</span>
                      <span>•</span>
                      <span>{citation.year}</span>
                    </div>
                    <Badge variant="neutral" size="sm">
                      {citation.client}
                    </Badge>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}
        </div>
      </div>
    </div>);

}