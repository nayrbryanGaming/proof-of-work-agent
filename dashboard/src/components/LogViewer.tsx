'use client';

import { useState, useMemo } from 'react';
import { FileText, RefreshCw, Filter, Download, Search } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface LogEntry {
  timestamp: string;
  level: string;
  module: string;
  message: string;
}

interface LogViewerProps {
  logs: LogEntry[];
  onRefresh: () => void;
}

export function LogViewer({ logs, onRefresh }: LogViewerProps) {
  const [filter, setFilter] = useState<string>('');
  const [levelFilter, setLevelFilter] = useState<string>('ALL');
  const [moduleFilter, setModuleFilter] = useState<string>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);

  const modules = useMemo(() => {
    const uniqueModules = new Set(logs.map(l => l.module));
    return ['ALL', ...Array.from(uniqueModules).sort()];
  }, [logs]);

  const filteredLogs = useMemo(() => {
    return logs.filter(log => {
      if (levelFilter !== 'ALL' && log.level !== levelFilter) return false;
      if (moduleFilter !== 'ALL' && log.module !== moduleFilter) return false;
      if (filter && !log.message.toLowerCase().includes(filter.toLowerCase())) return false;
      return true;
    });
  }, [logs, filter, levelFilter, moduleFilter]);

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'DEBUG': return 'text-gray-400';
      case 'INFO': return 'text-blue-400';
      case 'WARNING': return 'text-yellow-400';
      case 'ERROR': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getLevelBadgeColor = (level: string) => {
    switch (level) {
      case 'DEBUG': return 'bg-gray-500/20 text-gray-400';
      case 'INFO': return 'bg-blue-500/20 text-blue-400';
      case 'WARNING': return 'bg-yellow-500/20 text-yellow-400';
      case 'ERROR': return 'bg-red-500/20 text-red-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  const exportLogs = () => {
    const content = filteredLogs
      .map(l => `[${l.timestamp}][${l.level}][${l.module}] ${l.message}`)
      .join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agent-logs-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-solana-purple" />
          Agent Logs
        </h2>
        
        <div className="flex items-center gap-2">
          <button
            onClick={exportLogs}
            className="btn-secondary flex items-center gap-2 text-sm"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
          <button
            onClick={onRefresh}
            className="btn-secondary flex items-center gap-2 text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input
            type="text"
            placeholder="Search logs..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-900/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-primary-500"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-500" />
          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value)}
            className="px-3 py-2 bg-gray-900/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-primary-500"
          >
            <option value="ALL">All Levels</option>
            <option value="DEBUG">DEBUG</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
          </select>
          
          <select
            value={moduleFilter}
            onChange={(e) => setModuleFilter(e.target.value)}
            className="px-3 py-2 bg-gray-900/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-primary-500"
          >
            {modules.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <label className="flex items-center gap-2 text-sm text-gray-400">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="rounded border-gray-700 bg-gray-900 text-primary-500 focus:ring-primary-500"
          />
          Auto-scroll
        </label>
      </div>

      {/* Log Display */}
      <div className="log-terminal">
        {filteredLogs.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            No logs found
          </div>
        ) : (
          filteredLogs.map((log, index) => (
            <div key={index} className="log-entry flex items-start gap-2 text-sm">
              <span className="text-gray-600 shrink-0 w-32 truncate" title={log.timestamp}>
                {log.timestamp}
              </span>
              <span className={`shrink-0 px-2 py-0.5 rounded text-xs font-medium ${getLevelBadgeColor(log.level)}`}>
                {log.level}
              </span>
              <span className="shrink-0 text-gray-500 w-24 truncate" title={log.module}>
                [{log.module}]
              </span>
              <span className={`${getLevelColor(log.level)} break-all`}>
                {log.message}
              </span>
            </div>
          ))
        )}
      </div>

      <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
        <span>Showing {filteredLogs.length} of {logs.length} entries</span>
        <span>Last updated: {new Date().toLocaleTimeString()}</span>
      </div>
    </div>
  );
}
