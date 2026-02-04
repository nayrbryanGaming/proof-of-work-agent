'use client';

import { useState, useEffect } from 'react';
import { Settings, Save, RefreshCw, ExternalLink } from 'lucide-react';

interface Config {
  colosseum_base_url: string;
  solana_rpc: string;
  program_id: string;
  loop_interval: number;
  log_level: string;
  heartbeat_url: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function ConfigPanel() {
  const [config, setConfig] = useState<Config | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchConfig = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/config`);
      if (response.ok) {
        const data = await response.json();
        setConfig(data);
      }
    } catch (error) {
      console.error('Failed to fetch config:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchConfig();
  }, []);

  if (!config) {
    return (
      <div className="card">
        <div className="text-center text-gray-500 py-8">
          Loading configuration...
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Settings className="w-5 h-5 text-solana-purple" />
          Configuration
        </h2>
        
        <button
          onClick={fetchConfig}
          className="btn-secondary flex items-center gap-2 text-sm"
          disabled={isLoading}
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Colosseum Settings */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
            Colosseum API
          </h3>
          
          <ConfigItem
            label="Base URL"
            value={config.colosseum_base_url}
            type="url"
          />
          
          <ConfigItem
            label="Heartbeat URL"
            value={config.heartbeat_url}
            type="url"
          />
        </div>

        {/* Solana Settings */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
            Solana
          </h3>
          
          <ConfigItem
            label="RPC Endpoint"
            value={config.solana_rpc}
            type="url"
          />
          
          <ConfigItem
            label="Program ID"
            value={config.program_id}
            type="address"
            explorerUrl={`https://explorer.solana.com/address/${config.program_id}?cluster=devnet`}
          />
        </div>

        {/* Agent Settings */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
            Agent
          </h3>
          
          <ConfigItem
            label="Loop Interval"
            value={`${config.loop_interval} seconds`}
          />
          
          <ConfigItem
            label="Log Level"
            value={config.log_level}
          />
        </div>

        {/* Environment Info */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
            Environment
          </h3>
          
          <ConfigItem
            label="API URL"
            value={API_BASE}
            type="url"
          />
          
          <ConfigItem
            label="Dashboard Version"
            value="1.0.0"
          />
        </div>
      </div>

      {/* Info Box */}
      <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
        <h4 className="text-sm font-medium text-blue-400 mb-2">Configuration Note</h4>
        <p className="text-sm text-gray-400">
          To modify configuration, update the <code className="px-1.5 py-0.5 bg-gray-800 rounded text-solana-green">.env</code> file 
          in the backend and restart the API server. Sensitive values like API keys are not displayed here.
        </p>
      </div>
    </div>
  );
}

interface ConfigItemProps {
  label: string;
  value: string;
  type?: 'text' | 'url' | 'address';
  explorerUrl?: string;
}

function ConfigItem({ label, value, type = 'text', explorerUrl }: ConfigItemProps) {
  const isNotSet = value === 'NOT_SET' || !value;

  return (
    <div className="p-3 bg-gray-900/50 rounded-lg">
      <label className="block text-xs text-gray-500 mb-1">{label}</label>
      <div className="flex items-center justify-between gap-2">
        <span 
          className={`font-mono text-sm truncate ${isNotSet ? 'text-red-400' : 'text-white'}`}
          title={value}
        >
          {isNotSet ? 'NOT SET' : value}
        </span>
        
        {explorerUrl && !isNotSet && (
          <a
            href={explorerUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="p-1 text-gray-500 hover:text-solana-green transition-colors"
            title="View on Solana Explorer"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
      </div>
    </div>
  );
}
