import { useState } from 'react';
import { ChevronDown, ChevronRight, Circle, Clock, CheckCircle2 } from 'lucide-react';
import { statusColor, statusLabel } from '../utils/helpers';

const statusOptions = ['not_started', 'in_progress', 'completed'];

const statusIcons = {
  not_started: Circle,
  in_progress: Clock,
  completed: CheckCircle2,
};

export default function TopicItem({ topic, onStatusChange, onSelect, selectedTopicId, depth = 0 }) {
  const [expanded, setExpanded] = useState(true);
  const [updating, setUpdating] = useState(false);
  const hasChildren = topic.children && topic.children.length > 0;
  const Icon = statusIcons[topic.status] || Circle;

  const handleStatusChange = async (newStatus) => {
    if (newStatus === topic.status || updating) return;
    setUpdating(true);
    try {
      await onStatusChange(topic.id, newStatus);
    } finally {
      setUpdating(false);
    }
  };

  const cycleStatus = () => {
    const currentIdx = statusOptions.indexOf(topic.status);
    const nextIdx = (currentIdx + 1) % statusOptions.length;
    handleStatusChange(statusOptions[nextIdx]);
  };

  return (
    <div className={depth > 0 ? 'ml-6 border-l border-slate-700/50 pl-4' : ''}>
      <div
        className={`flex items-center gap-3 py-3 px-4 rounded-lg hover:bg-slate-800/50 transition-colors
                     ${updating ? 'opacity-60' : ''}`}
      >
        {hasChildren ? (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-slate-400 hover:text-slate-200 transition-colors"
          >
            {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </button>
        ) : (
          <span className="w-4" />
        )}

        <button
          onClick={cycleStatus}
          disabled={updating}
          className="flex-shrink-0 hover:scale-110 transition-transform"
          title={`Click to change status (current: ${statusLabel(topic.status)})`}
        >
          <Icon
            className={`w-5 h-5 ${
              topic.status === 'completed'
                ? 'text-emerald-400'
                : topic.status === 'in_progress'
                ? 'text-amber-400'
                : 'text-slate-500'
            }`}
          />
        </button>

        <button
          onClick={() => onSelect(topic)}
          className={`flex-1 text-sm text-left transition-colors
            ${selectedTopicId === topic.id
              ? 'text-indigo-400 font-medium'
              : 'text-slate-200 hover:text-indigo-300'}`}
        >
          {topic.title}
        </button>

        <select
          value={topic.status}
          onChange={(e) => handleStatusChange(e.target.value)}
          disabled={updating}
          className={`text-xs font-medium px-2.5 py-1 rounded-full border cursor-pointer
                      appearance-none text-center ${statusColor(topic.status)}
                      bg-transparent focus:outline-none focus:ring-2 focus:ring-indigo-500`}
        >
          {statusOptions.map((s) => (
            <option key={s} value={s} className="bg-slate-800 text-slate-200">
              {statusLabel(s)}
            </option>
          ))}
        </select>
      </div>

      {hasChildren && expanded && (
        <div className="mt-1">
          {topic.children.map((sub) => (
            <TopicItem
              key={sub.id}
              topic={sub}
              onStatusChange={onStatusChange}
              onSelect={onSelect}
              selectedTopicId={selectedTopicId}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}
