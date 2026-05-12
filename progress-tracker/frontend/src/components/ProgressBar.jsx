export default function ProgressBar({ percentage = 0, size = 'md', showLabel = true, className = '' }) {
  const height = size === 'sm' ? 'h-1.5' : size === 'lg' ? 'h-4' : 'h-2.5';
  const color =
    percentage >= 80 ? 'bg-emerald-500' : percentage >= 40 ? 'bg-indigo-500' : 'bg-amber-500';

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className={`flex-1 bg-slate-700/50 rounded-full ${height} overflow-hidden`}>
        <div
          className={`${height} ${color} rounded-full transition-all duration-700 ease-out`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-sm font-medium text-slate-300 min-w-[3rem] text-right">
          {percentage}%
        </span>
      )}
    </div>
  );
}
