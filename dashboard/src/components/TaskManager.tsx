'use client';

import { useState, useEffect } from 'react';
import { Hash, Plus, Trash2, RefreshCw } from 'lucide-react';

interface Task {
  id: number;
  description: string;
  category: string;
  difficulty: string;
  created_at?: string;
}

interface TaskManagerProps {
  onRefresh: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function TaskManager({ onRefresh }: TaskManagerProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [newTask, setNewTask] = useState({
    description: '',
    category: 'general',
    difficulty: 'medium',
  });

  const fetchTasks = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/tasks`);
      if (response.ok) {
        const data = await response.json();
        setTasks(data);
      }
    } catch (error) {
      console.error('Failed to fetch tasks:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTasks();
  }, []);

  const handleCreate = async () => {
    if (!newTask.description.trim()) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTask),
      });

      if (response.ok) {
        setNewTask({ description: '', category: 'general', difficulty: 'medium' });
        setShowForm(false);
        await fetchTasks();
      }
    } catch (error) {
      console.error('Failed to create task:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (taskId: number) => {
    if (!confirm('Are you sure you want to delete this task?')) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/tasks/${taskId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        await fetchTasks();
      }
    } catch (error) {
      console.error('Failed to delete task:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'bg-green-500/20 text-green-400';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400';
      case 'hard': return 'bg-red-500/20 text-red-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      solana: 'bg-solana-purple/20 text-solana-purple',
      defi: 'bg-blue-500/20 text-blue-400',
      ai: 'bg-purple-500/20 text-purple-400',
      security: 'bg-red-500/20 text-red-400',
      development: 'bg-cyan-500/20 text-cyan-400',
      architecture: 'bg-orange-500/20 text-orange-400',
      integration: 'bg-pink-500/20 text-pink-400',
      devops: 'bg-emerald-500/20 text-emerald-400',
    };
    return colors[category] || 'bg-gray-500/20 text-gray-400';
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Hash className="w-5 h-5 text-solana-green" />
          Task Queue
        </h2>
        
        <div className="flex items-center gap-2">
          <button
            onClick={fetchTasks}
            className="btn-secondary flex items-center gap-2 text-sm"
            disabled={isLoading}
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={() => setShowForm(!showForm)}
            className="btn-primary flex items-center gap-2 text-sm"
          >
            <Plus className="w-4 h-4" />
            Add Task
          </button>
        </div>
      </div>

      {/* Create Form */}
      {showForm && (
        <div className="mb-6 p-4 bg-gray-900/50 rounded-lg border border-gray-700">
          <h3 className="text-sm font-medium text-white mb-3">New Task</h3>
          
          <div className="space-y-3">
            <textarea
              value={newTask.description}
              onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
              placeholder="Task description (min 10 characters)..."
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-primary-500 resize-none"
              rows={3}
            />
            
            <div className="flex gap-4">
              <div className="flex-1">
                <label className="block text-xs text-gray-500 mb-1">Category</label>
                <select
                  value={newTask.category}
                  onChange={(e) => setNewTask({ ...newTask, category: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-primary-500"
                >
                  <option value="general">General</option>
                  <option value="solana">Solana</option>
                  <option value="defi">DeFi</option>
                  <option value="ai">AI</option>
                  <option value="security">Security</option>
                  <option value="development">Development</option>
                  <option value="architecture">Architecture</option>
                  <option value="integration">Integration</option>
                  <option value="devops">DevOps</option>
                </select>
              </div>
              
              <div className="flex-1">
                <label className="block text-xs text-gray-500 mb-1">Difficulty</label>
                <select
                  value={newTask.difficulty}
                  onChange={(e) => setNewTask({ ...newTask, difficulty: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-primary-500"
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
              </div>
            </div>
            
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowForm(false)}
                className="btn-secondary text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                className="btn-primary text-sm"
                disabled={newTask.description.length < 10}
              >
                Create Task
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Task List */}
      <div className="space-y-3">
        {tasks.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            No tasks in queue
          </div>
        ) : (
          tasks.map((task) => (
            <div
              key={task.id}
              className="p-4 bg-gray-900/50 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs text-gray-500">#{task.id}</span>
                    <span className={`px-2 py-0.5 text-xs rounded-full ${getCategoryColor(task.category)}`}>
                      {task.category}
                    </span>
                    <span className={`px-2 py-0.5 text-xs rounded-full ${getDifficultyColor(task.difficulty)}`}>
                      {task.difficulty}
                    </span>
                  </div>
                  <p className="text-white text-sm">{task.description}</p>
                </div>
                
                <button
                  onClick={() => handleDelete(task.id)}
                  className="p-2 text-gray-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                  title="Delete task"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="mt-4 text-sm text-gray-500">
        {tasks.length} task{tasks.length !== 1 ? 's' : ''} in queue
      </div>
    </div>
  );
}
