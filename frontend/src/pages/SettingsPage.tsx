import React, { useState } from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Toggle } from '../components/ui/Toggle';
import { Select } from '../components/ui/Select';
import { rebuildIndex } from '../api';

export function SettingsPage() {
  const [rebuildStatus, setRebuildStatus] = useState<'idle' | 'loading' | 'ok' | 'error'>('idle');
  const [rebuildMessage, setRebuildMessage] = useState('');

  const handleRebuildIndex = async () => {
    setRebuildStatus('loading');
    setRebuildMessage('');
    try {
      const r = await rebuildIndex();
      setRebuildStatus('ok');
      setRebuildMessage(r.message || 'Index cleared. Run a new query to rebuild with DOI/PMC ID.');
    } catch (e) {
      setRebuildStatus('error');
      setRebuildMessage((e as Error).message);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="text-2xl font-bold text-gray-900">Settings</h1>

      <Card>
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Organization Profile
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Input label="Organization Name" defaultValue="Consortium A" />
          <Input label="Contact Email" defaultValue="admin@consortium-a.org" />
          <Select
            label="Role"
            options={[
            {
              value: 'admin',
              label: 'Administrator'
            },
            {
              value: 'researcher',
              label: 'Researcher'
            },
            {
              value: 'clinician',
              label: 'Clinician'
            }]
            } />

        </div>
      </Card>

      <Card>
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Privacy Parameters
        </h3>
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Strict Privacy Mode</p>
              <p className="text-sm text-gray-500">
                Enforce maximum noise levels for all queries
              </p>
            </div>
            <Toggle checked={true} onChange={() => {}} />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Audit Logging</p>
              <p className="text-sm text-gray-500">
                Log all query metadata for compliance
              </p>
            </div>
            <Toggle checked={true} onChange={() => {}} />
          </div>
          <div className="grid grid-cols-2 gap-6 pt-4 border-t border-gray-100">
            <Input label="Default Epsilon (ε)" defaultValue="1.0" />
            <Input label="Default Delta (δ)" defaultValue="1e-5" />
          </div>
        </div>
      </Card>

      <Card>
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Index &amp; Citations (DOI / PMC ID)
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          If retrieval results show empty &quot;DOI / ID&quot;, the vector index was built before
          identifiers were added. Rebuild the index so citations include DOI or PMC ID from your dataset.
        </p>
        <Button
          onClick={handleRebuildIndex}
          disabled={rebuildStatus === 'loading'}
        >
          {rebuildStatus === 'loading' ? 'Rebuilding…' : 'Rebuild index (to show DOI/PMC ID)'}
        </Button>
        {rebuildStatus === 'ok' && (
          <p className="mt-3 text-sm text-green-700">{rebuildMessage}</p>
        )}
        {rebuildStatus === 'error' && (
          <p className="mt-3 text-sm text-red-600">{rebuildMessage}</p>
        )}
      </Card>

      <Card>
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          API Configuration
        </h3>
        <div className="space-y-4">
          <Input
            label="OpenRouter API Key"
            type="password"
            defaultValue="sk-........................" />

          <Input
            label="ChromaDB Endpoint"
            defaultValue="https://db.consortium-a.org:8000" />

          <div className="flex justify-end mt-4">
            <Button>Save Changes</Button>
          </div>
        </div>
      </Card>
    </div>);

}