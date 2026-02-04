'use client';

import { formatDistanceToNow } from 'date-fns';
import { CheckCircle, XCircle, Clock, Hash, Wallet, MessageSquare, Activity } from 'lucide-react';

interface CycleResult {
  cycle_number: number;
  heartbeat_synced: boolean;
  status_checked: boolean;
  forum_engaged: boolean;
  task_solved: boolean;
  task_hash: string | null;
  solana_tx: string | null;
  project_updated: boolean;
  duration: number;
  errors: string[];
  timestamp: string;
}

interface CycleHistoryProps {
  cycles: CycleResult[];
}

export function CycleHistory({ cycles }: CycleHistoryProps) {
  if (cycles.length === 0) {
    return (
      <div className="card">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-solana-purple" />
          Recent Cycles
        </h2>
        <div className="text-center text-gray-500 py-8">
          No cycles recorded yet
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-solana-purple" />
        Recent Cycles
      </h2>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-sm text-gray-500 border-b border-gray-700">
              <th className="pb-3 font-medium">#</th>
              <th className="pb-3 font-medium">Time</th>
              <th className="pb-3 font-medium">Heartbeat</th>
              <th className="pb-3 font-medium">Forum</th>
              <th className="pb-3 font-medium">Task</th>
              <th className="pb-3 font-medium">Solana TX</th>
              <th className="pb-3 font-medium">Duration</th>
              <th className="pb-3 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {cycles.map((cycle) => (
              <CycleRow key={cycle.cycle_number} cycle={cycle} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CycleRow({ cycle }: { cycle: CycleResult }) {
  const hasErrors = cycle.errors && cycle.errors.length > 0;
  
  const formatTime = (isoString: string) => {
    try {
      return formatDistanceToNow(new Date(isoString), { addSuffix: true });
    } catch {
      return 'Unknown';
    }
  };

  const StatusIcon = ({ active }: { active: boolean }) => (
    active ? (
      <CheckCircle className="w-4 h-4 text-solana-green" />
    ) : (
      <XCircle className="w-4 h-4 text-gray-600" />
    )
  );

  return (
    <tr className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
      <td className="py-3 text-white font-medium">
        #{cycle.cycle_number}
      </td>
      <td className="py-3 text-gray-400 text-sm">
        {formatTime(cycle.timestamp)}
      </td>
      <td className="py-3">
        <StatusIcon active={cycle.heartbeat_synced} />
      </td>
      <td className="py-3">
        <StatusIcon active={cycle.forum_engaged} />
      </td>
      <td className="py-3">
        <div className="flex items-center gap-2">
          <StatusIcon active={cycle.task_solved} />
          {cycle.task_hash && (
            <span className="text-xs text-gray-500 font-mono" title={cycle.task_hash}>
              {cycle.task_hash.slice(0, 8)}...
            </span>
          )}
        </div>
      </td>
      <td className="py-3">
        {cycle.solana_tx ? (
          <a
            href={`https://explorer.solana.com/tx/${cycle.solana_tx}?cluster=devnet`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-solana-green hover:underline font-mono"
          >
            {cycle.solana_tx.slice(0, 8)}...
          </a>
        ) : (
          <span className="text-gray-600">-</span>
        )}
      </td>
      <td className="py-3 text-gray-400 text-sm">
        {cycle.duration.toFixed(2)}s
      </td>
      <td className="py-3">
        {hasErrors ? (
          <span className="px-2 py-1 bg-red-500/20 text-red-400 text-xs rounded-full">
            {cycle.errors.length} error{cycle.errors.length > 1 ? 's' : ''}
          </span>
        ) : (
          <span className="px-2 py-1 bg-solana-green/20 text-solana-green text-xs rounded-full">
            Success
          </span>
        )}
      </td>
    </tr>
  );
}
