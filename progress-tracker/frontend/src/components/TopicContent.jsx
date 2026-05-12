import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { X, Loader2 } from 'lucide-react';
import client from '../api/client';

export default function TopicContent({ topicId, topicTitle, onClose }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchContent = async () => {
      setLoading(true);
      try {
        const res = await client.get(`/projects/topics/${topicId}`);
        setContent(res.data.content);
      } catch (err) {
        console.error('Failed to load topic content:', err);
        setContent(null);
      } finally {
        setLoading(false);
      }
    };
    fetchContent();
  }, [topicId]);

  return (
    <div className="card border border-slate-700/50 mt-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-100">{topicTitle}</h3>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-slate-200 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

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
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      ) : (
        <p className="text-slate-500 text-sm italic py-8 text-center">
          No content available for this topic yet.
        </p>
      )}
    </div>
  );
}
