import { useState } from 'react';
import { CheckCircle2, Circle } from 'lucide-react';
import client from '../api/client';

export default function CheckpointButton({ topicId, checkpointNumber, label, confirmed: initialConfirmed, onConfirm }) {
  const [confirmed, setConfirmed] = useState(initialConfirmed);
  const [loading, setLoading] = useState(false);

  const handleConfirm = async () => {
    if (confirmed || loading) return;
    setLoading(true);
    try {
      await client.post(`/projects/topics/${topicId}/checkpoints/${checkpointNumber}/confirm`);
      setConfirmed(true);
      onConfirm?.(checkpointNumber);
    } catch (err) {
      console.error('Failed to confirm checkpoint:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`my-6 mx-0 flex items-center gap-3 p-4 rounded-xl border transition-all duration-300
      ${confirmed
        ? 'bg-emerald-950/30 border-emerald-700/50'
        : 'bg-indigo-950/30 border-indigo-700/50 hover:border-indigo-500/70 cursor-pointer'
      }`}
      onClick={handleConfirm}
    >
      {confirmed ? (
        <CheckCircle2 className="w-6 h-6 text-emerald-400 flex-shrink-0" />
      ) : (
        <Circle className={`w-6 h-6 flex-shrink-0 ${loading ? 'text-indigo-300 animate-pulse' : 'text-indigo-400'}`} />
      )}
      <div className="flex-1 min-w-0">
        <span className={`text-sm font-medium ${confirmed ? 'text-emerald-300' : 'text-indigo-200'}`}>
          {confirmed ? '✓ Confirmed' : 'Checkpoint'}
        </span>
        <p className={`text-sm mt-0.5 ${confirmed ? 'text-emerald-400/70' : 'text-slate-300'}`}>
          {label}
        </p>
      </div>
      {!confirmed && (
        <span className="text-xs text-indigo-400 bg-indigo-900/50 px-2.5 py-1 rounded-full whitespace-nowrap">
          Click to confirm
        </span>
      )}
    </div>
  );
}
