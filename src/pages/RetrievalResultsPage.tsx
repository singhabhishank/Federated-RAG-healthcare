import React, { useState, Children } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { Tooltip } from '../components/ui/Tooltip';
import {
  ArrowRight,
  Filter,
  Info,
  ChevronDown,
  FileText,
  Calendar,
  Building,
  Lock } from
'lucide-react';
const mockResults = [
{
  id: 1,
  title:
  'Efficacy of ACE Inhibitors in Heart Failure Management: A Multi-Center Study',
  journal: 'Journal of Cardiology',
  year: 2024,
  client: 'Client 0',
  relevance: 92,
  included: true,
  authors: 'Smith J., Doe A., et al.'
},
{
  id: 2,
  title: 'Long-term Outcomes of Beta-Blocker Therapy in Elderly Patients',
  journal: 'Geriatric Medicine',
  year: 2023,
  client: 'Client 1',
  relevance: 88,
  included: true,
  authors: 'Johnson L., Brown K.'
},
{
  id: 3,
  title: 'Comparative Analysis of ARBs vs ACE Inhibitors',
  journal: 'Heart Health Weekly',
  year: 2025,
  client: 'Client 2',
  relevance: 85,
  included: true,
  authors: 'Williams R., et al.'
},
{
  id: 4,
  title: 'Side Effects Profile of Modern Hypertension Medications',
  journal: 'Clinical Pharmacology',
  year: 2022,
  client: 'Client 0',
  relevance: 76,
  included: false,
  authors: 'Davis M., Wilson T.'
},
{
  id: 5,
  title: 'Patient Adherence to Heart Failure Regimens',
  journal: 'Nursing Research',
  year: 2023,
  client: 'Client 1',
  relevance: 72,
  included: false,
  authors: 'Miller P., Taylor S.'
}];

export function RetrievalResultsPage() {
  const [expandedId, setExpandedId] = useState<number | null>(null);
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
            Aggregated from 3 secure nodes • 15 results found
          </p>
        </div>
        <Link to="/answers">
          <Button rightIcon={<ArrowRight className="w-4 h-4" />}>
            Generate Answer
          </Button>
        </Link>
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
        </div>
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-wrap gap-4 items-end">
          <div className="w-full sm:w-48">
            <Input label="Year Range" placeholder="2020 - 2026" />
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

      {/* Results List */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-4">

        {mockResults.map((result) =>
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

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-gray-700">
                        Authors:
                      </span>
                      <p className="text-gray-600">{result.authors}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">DOI:</span>
                      <p className="text-gray-600 font-mono">
                        10.1056/NEJMoa240123
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