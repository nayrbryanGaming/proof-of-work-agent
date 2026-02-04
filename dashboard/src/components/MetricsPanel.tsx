'use client';

import { useMemo } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { TrendingUp, Clock, CheckCircle, XCircle, MessageSquare, Wallet } from 'lucide-react';

interface Metrics {
  uptime_seconds: number;
  total_cycles: number;
  successful_cycles: number;
  failed_cycles: number;
  tasks_solved: number;
  forum_engagements: number;
  solana_transactions: number;
  average_cycle_duration: number;
  error_rate: number;
}

interface MetricsPanelProps {
  metrics: Metrics | null;
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  if (!metrics) {
    return (
      <div className="card">
        <div className="text-center text-gray-500 py-8">
          Loading metrics...
        </div>
      </div>
    );
  }

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const successRate = metrics.total_cycles > 0 
    ? ((metrics.successful_cycles / metrics.total_cycles) * 100).toFixed(1)
    : '0';

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-solana-green" />
        Metrics
      </h2>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <MetricCard
          label="Uptime"
          value={formatUptime(metrics.uptime_seconds)}
          icon={<Clock className="w-4 h-4" />}
          color="blue"
        />
        <MetricCard
          label="Total Cycles"
          value={String(metrics.total_cycles)}
          icon={<TrendingUp className="w-4 h-4" />}
          color="purple"
        />
        <MetricCard
          label="Success Rate"
          value={`${successRate}%`}
          icon={<CheckCircle className="w-4 h-4" />}
          color="green"
        />
        <MetricCard
          label="Tasks Solved"
          value={String(metrics.tasks_solved)}
          icon={<CheckCircle className="w-4 h-4" />}
          color="emerald"
        />
        <MetricCard
          label="Forum Engagements"
          value={String(metrics.forum_engagements)}
          icon={<MessageSquare className="w-4 h-4" />}
          color="cyan"
        />
        <MetricCard
          label="Solana TXs"
          value={String(metrics.solana_transactions)}
          icon={<Wallet className="w-4 h-4" />}
          color="violet"
        />
      </div>

      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <h3 className="text-sm text-gray-400 mb-2">Avg Cycle Duration</h3>
          <p className="text-2xl font-semibold text-white">
            {metrics.average_cycle_duration.toFixed(2)}s
          </p>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <h3 className="text-sm text-gray-400 mb-2">Error Rate</h3>
          <p className={`text-2xl font-semibold ${metrics.error_rate > 10 ? 'text-red-500' : 'text-solana-green'}`}>
            {metrics.error_rate.toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  color: 'blue' | 'purple' | 'green' | 'emerald' | 'cyan' | 'violet';
}

function MetricCard({ label, value, icon, color }: MetricCardProps) {
  const colorClasses = {
    blue: 'text-blue-400 bg-blue-500/10',
    purple: 'text-purple-400 bg-purple-500/10',
    green: 'text-solana-green bg-solana-green/10',
    emerald: 'text-emerald-400 bg-emerald-500/10',
    cyan: 'text-cyan-400 bg-cyan-500/10',
    violet: 'text-violet-400 bg-violet-500/10',
  };

  return (
    <div className="bg-gray-900/50 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1">
        <span className={`p-1 rounded ${colorClasses[color]}`}>
          {icon}
        </span>
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <p className="text-lg font-semibold text-white">{value}</p>
    </div>
  );
}
