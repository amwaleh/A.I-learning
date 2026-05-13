import { CheckCircle2, BookOpen, Clock } from 'lucide-react';

export default function ProgressIndicator({ scrollPercent, timeSpent, checkpointsTotal, checkpointsConfirmed, status }) {
  const allCheckpointsDone = checkpointsConfirmed >= checkpointsTotal;
  const isComplete = status === 'completed';
  const timeFormatted = timeSpent >= 60 ? `${Math.floor(timeSpent / 60)}m ${timeSpent % 60}s` : `${timeSpent}s`;

  if (isComplete) {
    return (
      <div className="flex items-center gap-2 px-4 py-2.5 bg-emerald-950/40 border-t border-emerald-800/50 text-emerald-300 text-sm">
        <CheckCircle2 className="w-4 h-4" />
        <span className="font-medium">Topic completed</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4 px-4 py-2.5 bg-slate-900/80 border-t border-slate-800 text-sm">
      {/* Scroll progress */}
      <div className="flex items-center gap-1.5 text-slate-400">
        <BookOpen className="w-3.5 h-3.5" />
        <span>{scrollPercent}% read</span>
      </div>

      {/* Checkpoints */}
      {checkpointsTotal > 0 && (
        <div className={`flex items-center gap-1.5 ${allCheckpointsDone ? 'text-emerald-400' : 'text-indigo-400'}`}>
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>{checkpointsConfirmed}/{checkpointsTotal} checkpoints</span>
        </div>
      )}

      {/* Time */}
      <div className="flex items-center gap-1.5 text-slate-500 ml-auto">
        <Clock className="w-3.5 h-3.5" />
        <span>{timeFormatted}</span>
      </div>

      {/* Progress bar */}
      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all duration-500"
          style={{ width: `${scrollPercent}%` }}
        />
      </div>
    </div>
  );
}
