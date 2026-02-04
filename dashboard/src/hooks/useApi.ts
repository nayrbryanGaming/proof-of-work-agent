'use client';

import { useState, useCallback, useEffect } from 'react';

interface AgentStatus {
  running: boolean;
  cycle_count: number;
  last_cycle_at: string | null;
  last_heartbeat_at: string | null;
  last_forum_at: string | null;
  last_solana_tx: string | null;
  errors_count: number;
  tasks_solved: number;
}

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

interface LogEntry {
  timestamp: string;
  level: string;
  module: string;
  message: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export function useApi() {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [cycles, setCycles] = useState<CycleResult[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [statusData, metricsData, cyclesData, logsData] = await Promise.all([
        fetchApi<AgentStatus>('/api/status'),
        fetchApi<Metrics>('/api/metrics'),
        fetchApi<CycleResult[]>('/api/cycles?limit=20'),
        fetchApi<LogEntry[]>('/api/logs?limit=100'),
      ]);

      setStatus(statusData);
      setMetrics(metricsData);
      setCycles(cyclesData);
      setLogs(logsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const startAgent = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await fetchApi('/api/start', { method: 'POST' });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start agent');
    } finally {
      setIsLoading(false);
    }
  }, [refresh]);

  const stopAgent = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await fetchApi('/api/stop', { method: 'POST' });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop agent');
    } finally {
      setIsLoading(false);
    }
  }, [refresh]);

  const triggerCycle = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await fetchApi('/api/trigger-cycle', { method: 'POST' });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger cycle');
    } finally {
      setIsLoading(false);
    }
  }, [refresh]);

  // Initial fetch
  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    status,
    metrics,
    cycles,
    logs,
    isLoading,
    error,
    refresh,
    startAgent,
    stopAgent,
    triggerCycle,
  };
}
