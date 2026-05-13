import { useState, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { X, Loader2, PartyPopper } from 'lucide-react';
import client from '../api/client';
import MermaidBlock from './MermaidBlock';
import CheckpointButton from './CheckpointButton';

export default function TopicContent({ topicId, topicTitle, onClose, fullHeight = false, onStatusChange }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [checkpoints, setCheckpoints] = useState([]);
  const [status, setStatus] = useState('not_started');
  const [showCompleted, setShowCompleted] = useState(false);

  // Fetch content and checkpoints
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [contentRes, cpRes] = await Promise.all([
          client.get(`/projects/topics/${topicId}`),
          client.get(`/projects/topics/${topicId}/checkpoints`),
        ]);
        setContent(contentRes.data.content);
        setStatus(contentRes.data.status || 'not_started');
        setCheckpoints(cpRes.data || []);
      } catch (err) {
        console.error('Failed to load topic content:', err);
        setContent(null);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [topicId]);

  // Reset overlay when topic changes
  useEffect(() => {
    setShowCompleted(false);
  }, [topicId]);

  // Handle status transition — show celebration before notifying parent
  const handleStatusTransition = useCallback((newStatus) => {
    if (newStatus === status) return;
    setStatus(newStatus);
    if (newStatus === 'completed') {
      setShowCompleted(true);
      setTimeout(() => {
        onStatusChange?.(topicId, newStatus);
      }, 3000);
    } else {
      onStatusChange?.(topicId, newStatus);
    }
  }, [status, topicId, onStatusChange]);

  const handleCheckpointConfirm = useCallback((cpNumber) => {
    setCheckpoints(prev =>
      prev.map(cp => cp.checkpoint_number === cpNumber ? { ...cp, confirmed: true } : cp)
    );
    // Check completion via API (pass scroll/time as met since user confirmed)
    client.patch(`/projects/topics/${topicId}/auto-progress`, {
      scroll_percent: 100,
      time_spent: 30,
    }).then((res) => {
      handleStatusTransition(res.data.status);
    }).catch(() => {});
  }, [topicId, handleStatusTransition]);

  // Parse checkpoint markers from content
  const renderContent = (rawContent) => {
    if (!rawContent) return rawContent;
    // Replace <!-- checkpoint: Label --> with a special placeholder
    return rawContent.replace(
      /<!--\s*checkpoint:\s*(.+?)\s*-->/g,
      (_, label) => `\n\n:::checkpoint:::${label}:::\n\n`
    );
  };

  const confirmedCount = checkpoints.filter(cp => cp.confirmed).length;

  return (
    <div className={`bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col relative
                     ${fullHeight ? 'h-full' : 'card mt-4'}`}>
      {/* Sticky header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 flex-shrink-0">
        <h3 className="text-lg font-semibold text-slate-100 truncate">{topicTitle}</h3>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-slate-200 transition-colors flex-shrink-0"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-6 h-6 text-indigo-500 animate-spin" />
          </div>
        ) : content ? (
          <div className="prose prose-invert prose-slate max-w-none
                          prose-headings:text-slate-100 prose-headings:font-semibold
                          prose-h2:text-xl prose-h2:mt-6 prose-h2:mb-3
                          prose-h3:text-lg prose-h3:mt-4 prose-h3:mb-2
                          prose-p:text-slate-300 prose-p:leading-relaxed
                          prose-li:text-slate-300
                          prose-strong:text-slate-100
                          prose-code:text-indigo-300 prose-code:bg-slate-800 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded
                          prose-pre:bg-slate-900 prose-pre:border prose-pre:border-slate-700
                          prose-a:text-indigo-400 prose-a:no-underline hover:prose-a:underline
                          prose-table:border-slate-700
                          prose-th:text-slate-200 prose-th:border-slate-600
                          prose-td:border-slate-700 prose-td:text-slate-300">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ className, children, ...props }) {
                  const match = /language-mermaid/.test(className || '');
                  if (match) {
                    return <MermaidBlock chart={String(children)} />;
                  }
                  return <code className={className} {...props}>{children}</code>;
                },
                pre({ children, ...props }) {
                  const child = Array.isArray(children) ? children[0] : children;
                  if (child?.props?.className === 'language-mermaid') {
                    return <>{children}</>;
                  }
                  return <pre {...props}>{children}</pre>;
                },
                p({ children }) {
                  // Detect checkpoint placeholder
                  const text = typeof children === 'string' ? children :
                    Array.isArray(children) ? children.join('') : '';
                  const cpMatch = String(text).match(/^:::checkpoint:::(.+?):::$/);
                  if (cpMatch) {
                    const label = cpMatch[1];
                    const cp = checkpoints.find(c => c.label === label);
                    if (cp) {
                      return (
                        <CheckpointButton
                          topicId={topicId}
                          checkpointNumber={cp.checkpoint_number}
                          label={label}
                          confirmed={cp.confirmed}
                          onConfirm={handleCheckpointConfirm}
                        />
                      );
                    }
                  }
                  return <p>{children}</p>;
                },
              }}
            >{renderContent(content)}</ReactMarkdown>
          </div>
        ) : (
          <p className="text-slate-500 text-sm italic py-8 text-center">
            No content available for this topic yet.
          </p>
        )}
      </div>

      {/* Checkpoint progress summary */}
      {!loading && content && checkpoints.length > 0 && (
        <div className="flex items-center gap-3 px-4 py-2.5 bg-slate-900/80 border-t border-slate-800 text-sm text-slate-400">
          <span>{confirmedCount}/{checkpoints.length} checkpoints confirmed</span>
          {status === 'completed' && <span className="text-emerald-400 font-medium">✓ Complete</span>}
        </div>
      )}

      {/* Completion overlay */}
      {showCompleted && (
        <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm rounded-xl flex items-center justify-center z-10 animate-fade-in">
          <div className="text-center space-y-3">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-emerald-500/20 mb-2">
              <PartyPopper className="w-8 h-8 text-emerald-400" />
            </div>
            <h3 className="text-xl font-bold text-emerald-300">Topic Completed!</h3>
            <p className="text-slate-400 text-sm">Moving to next topic...</p>
            <div className="w-32 mx-auto h-1 bg-slate-800 rounded-full overflow-hidden mt-4">
              <div className="h-full bg-emerald-500 rounded-full animate-progress-bar" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
