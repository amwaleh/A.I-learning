import { useNavigate } from 'react-router-dom';
import { FolderOpen } from 'lucide-react';
import ProgressBar from './ProgressBar';

const badgeColors = [
  'bg-indigo-500/20 text-indigo-400',
  'bg-emerald-500/20 text-emerald-400',
  'bg-amber-500/20 text-amber-400',
  'bg-rose-500/20 text-rose-400',
  'bg-cyan-500/20 text-cyan-400',
  'bg-purple-500/20 text-purple-400',
];

export default function ProjectCard({ project }) {
  const navigate = useNavigate();
  const badge = badgeColors[(project.number - 1) % badgeColors.length];

  return (
    <button
      onClick={() => navigate(`/projects/${project.id}`)}
      className="card text-left w-full hover:border-indigo-500/50 hover:bg-slate-800/80
                 transition-all duration-300 group cursor-pointer"
    >
      <div className="flex items-start justify-between mb-3">
        <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${badge}`}>
          #{project.number}
        </span>
        <FolderOpen className="w-5 h-5 text-slate-500 group-hover:text-indigo-400 transition-colors" />
      </div>

      <h3 className="text-lg font-semibold text-slate-100 mb-1.5 group-hover:text-indigo-300 transition-colors">
        {project.title}
      </h3>

      {project.description && (
        <p className="text-sm text-slate-400 mb-4 line-clamp-2">{project.description}</p>
      )}

      <div className="mt-auto">
        <div className="flex justify-between text-xs text-slate-400 mb-1.5">
          <span>
            {project.completed_topics} / {project.total_topics} topics
          </span>
          <span>{project.progress_percentage}%</span>
        </div>
        <ProgressBar percentage={project.progress_percentage} showLabel={false} size="sm" />
      </div>
    </button>
  );
}
