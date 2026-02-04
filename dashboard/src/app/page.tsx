'use client';

import { useState, useEffect, useCallback } from 'react';
import { 
  Activity, 
  Play, 
  Square, 
  RefreshCw, 
  Cpu, 
  MessageSquare, 
  Hash, 
  Wallet,
  Clock,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Settings,
  FileText
} from 'lucide-react';
import { MetricsPanel } from '@/components/MetricsPanel';
import { LogViewer } from '@/components/LogViewer';
import { CycleHistory } from '@/components/CycleHistory';
import { TaskManager } from '@/components/TaskManager';
import { ConfigPanel } from '@/components/ConfigPanel';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useApi } from '@/hooks/useApi';
import { formatDistanceToNow } from 'date-fns';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'logs' | 'tasks' | 'config'>('overview');
  const { status, metrics, cycles, logs, isLoading, error, refresh, startAgent, stopAgent, triggerCycle } = useApi();
  const { isConnected, lastMessage } = useWebSocket();
  
  // Auto-refresh data
  useEffect(() => {
    const interval = setInterval(() => {
      refresh();
    }, 10000);
    return () => clearInterval(interval);
  }, [refresh]);

  // Handle WebSocket updates
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'cycle_complete' || lastMessage.type === 'state_update') {
        refresh();
      }
    }
  }, [lastMessage, refresh]);

  const formatTime = (isoString: string | null) => {
    if (!isoString) return 'Never';
    try {
      return formatDistanceToNow(new Date(isoString), { addSuffix: true });
    } catch {
      return 'Unknown';
    }
  };

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-solana-purple to-solana-green flex items-center justify-center">
              <Cpu className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">POW Agent Dashboard</h1>
              <p className="text-gray-400 text-sm">Colosseum Solana Agent Hackathon</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Connection Status */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-lg">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-solana-green' : 'bg-red-500'}`} />
              <span className="text-sm text-gray-400">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {/* Agent Controls */}
            <div className="flex items-center gap-2">
              {status?.running ? (
                <button
                  onClick={stopAgent}
                  className="btn-danger flex items-center gap-2"
                  disabled={isLoading}
                >
                  <Square className="w-4 h-4" />
                  Stop Agent
                </button>
              ) : (
                <button
                  onClick={startAgent}
                  className="btn-primary flex items-center gap-2"
                  disabled={isLoading}
                >
                  <Play className="w-4 h-4" />
                  Start Agent
                </button>
              )}
              
              <button
                onClick={triggerCycle}
                className="btn-secondary flex items-center gap-2"
                disabled={isLoading || status?.running}
                title={status?.running ? 'Stop agent to trigger manual cycle' : 'Trigger single cycle'}
              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                Manual Cycle
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-500" />
          <span className="text-red-400">{error}</span>
        </div>
      )}

      {/* Navigation Tabs */}
      <nav className="mb-6">
        <div className="flex gap-2">
          {[
            { id: 'overview', label: 'Overview', icon: Activity },
            { id: 'logs', label: 'Logs', icon: FileText },
            { id: 'tasks', label: 'Tasks', icon: Hash },
            { id: 'config', label: 'Config', icon: Settings },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                activeTab === id
                  ? 'bg-primary-500/20 text-primary-400 border border-primary-500/50'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main>
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatusCard
                title="Agent Status"
                value={status?.running ? 'Running' : 'Stopped'}
                icon={<Activity className="w-5 h-5" />}
                status={status?.running ? 'success' : 'idle'}
                subtitle={`Cycle #${status?.cycle_count || 0}`}
              />
              <StatusCard
                title="Last Heartbeat"
                value={formatTime(status?.last_heartbeat_at || null)}
                icon={<Clock className="w-5 h-5" />}
                status={status?.last_heartbeat_at ? 'success' : 'idle'}
              />
              <StatusCard
                title="Tasks Solved"
                value={String(status?.tasks_solved || 0)}
                icon={<CheckCircle className="w-5 h-5" />}
                status="success"
              />
              <StatusCard
                title="Solana TX"
                value={status?.last_solana_tx ? `${status.last_solana_tx.slice(0, 8)}...` : 'None'}
                icon={<Wallet className="w-5 h-5" />}
                status={status?.last_solana_tx ? 'success' : 'idle'}
              />
            </div>

            {/* Metrics Panel */}
            <MetricsPanel metrics={metrics} />

            {/* Recent Cycles */}
            <CycleHistory cycles={cycles} />
          </div>
        )}

        {activeTab === 'logs' && <LogViewer logs={logs} onRefresh={refresh} />}
        {activeTab === 'tasks' && <TaskManager onRefresh={refresh} />}
        {activeTab === 'config' && <ConfigPanel />}
      </main>

      {/* Footer */}
      <footer className="mt-12 pt-6 border-t border-gray-800">
        <div className="flex items-center justify-between text-sm text-gray-500">
          <span>POW Agent v1.0.0</span>
          <span>Built for Colosseum Solana Agent Hackathon</span>
        </div>
      </footer>
    </div>
  );
}

interface StatusCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
  status: 'success' | 'error' | 'idle';
  subtitle?: string;
}

function StatusCard({ title, value, icon, status, subtitle }: StatusCardProps) {
  const statusColors = {
    success: 'text-solana-green',
    error: 'text-red-500',
    idle: 'text-yellow-500',
  };

  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-gray-400 text-sm">{title}</p>
          <p className={`text-xl font-semibold mt-1 ${statusColors[status]}`}>{value}</p>
          {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
        </div>
        <div className={`p-2 rounded-lg bg-gray-700/50 ${statusColors[status]}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}
