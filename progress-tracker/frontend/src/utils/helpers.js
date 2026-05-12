export function calcPercentage(completed, total) {
  if (!total || total === 0) return 0;
  return Math.round((completed / total) * 100);
}

export function formatDate(dateString) {
  if (!dateString) return '';
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

export function statusColor(status) {
  switch (status) {
    case 'completed':
      return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'in_progress':
      return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    default:
      return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
  }
}

export function statusLabel(status) {
  switch (status) {
    case 'completed':
      return 'Completed';
    case 'in_progress':
      return 'In Progress';
    default:
      return 'Not Started';
  }
}

export function progressBarColor(percentage) {
  if (percentage >= 80) return 'bg-emerald-500';
  if (percentage >= 40) return 'bg-indigo-500';
  return 'bg-amber-500';
}
