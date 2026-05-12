import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import client from '../api/client';
import ProjectCard from '../components/ProjectCard';
import ProgressBar from '../components/ProgressBar';
import { Flame, Target, FolderOpen, Loader2 } from 'lucide-react';

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [dashRes, projRes] = await Promise.all([
          client.get('/dashboard'),
          client.get('/projects'),
        ]);
        setStats(dashRes.data);
        setProjects(projRes.data);
      } catch (err) {
        console.error('Failed to load dashboard:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="card">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <h1 className="text-2xl font-bold text-slate-100">
              Welcome back, {user?.display_name || 'Learner'} 👋
            </h1>
            <p className="text-slate-400 mt-1">Keep up the great work on your learning journey!</p>
          </div>

          {stats?.streak_days > 0 && (
            <div className="flex items-center gap-2 px-4 py-2 bg-orange-500/10 border border-orange-500/30 rounded-xl">
              <Flame className="w-6 h-6 text-orange-400" />
              <span className="text-lg font-bold text-orange-400">{stats.streak_days} day streak</span>
            </div>
          )}
        </div>

        {stats && (
          <div className="mt-6">
            <div className="flex justify-between text-sm text-slate-400 mb-2">
              <span>Overall Progress</span>
              <span>{stats.overall_percentage}%</span>
            </div>
            <ProgressBar percentage={stats.overall_percentage} size="lg" showLabel={false} />
          </div>
        )}
      </div>

      {/* Stats Row */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="card flex items-center gap-4">
            <div className="p-3 bg-emerald-500/10 rounded-xl">
              <Target className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-slate-100">
                {stats.completed_topics}/{stats.total_topics}
              </p>
              <p className="text-sm text-slate-400">Topics Completed</p>
            </div>
          </div>

          <div className="card flex items-center gap-4">
            <div className="p-3 bg-indigo-500/10 rounded-xl">
              <FolderOpen className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-slate-100">{stats.projects_in_progress}</p>
              <p className="text-sm text-slate-400">Projects In Progress</p>
            </div>
          </div>

          <div className="card flex items-center gap-4">
            <div className="p-3 bg-orange-500/10 rounded-xl">
              <Flame className="w-6 h-6 text-orange-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-slate-100">{stats.streak_days || 0}</p>
              <p className="text-sm text-slate-400">Day Streak</p>
            </div>
          </div>
        </div>
      )}

      {/* Projects Grid */}
      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Your Projects</h2>
        {projects.length === 0 ? (
          <div className="card text-center py-12">
            <FolderOpen className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400">No projects yet. Start your learning journey!</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map((project) => (
              <ProjectCard key={project.id} project={project} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
