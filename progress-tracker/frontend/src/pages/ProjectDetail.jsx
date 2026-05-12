import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import client from '../api/client';
import TopicItem from '../components/TopicItem';
import ProgressBar from '../components/ProgressBar';
import { ArrowLeft, Loader2, BookOpen } from 'lucide-react';
import { calcPercentage } from '../utils/helpers';

export default function ProjectDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState(null);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [projRes, topicsRes] = await Promise.all([
        client.get('/projects'),
        client.get(`/projects/${id}/topics`),
      ]);
      const proj = projRes.data.find((p) => String(p.id) === String(id));
      setProject(proj);
      setTopics(topicsRes.data);
    } catch (err) {
      console.error('Failed to load project:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [id]);

  const handleStatusChange = async (topicId, newStatus) => {
    try {
      await client.patch(`/progress/${topicId}`, { status: newStatus });
      // Update topic in local state
      const updateTopicStatus = (items) =>
        items.map((t) => {
          if (t.id === topicId) return { ...t, status: newStatus };
          if (t.sub_topics) return { ...t, sub_topics: updateTopicStatus(t.sub_topics) };
          return t;
        });
      setTopics(updateTopicStatus(topics));

      // Refresh project stats
      const projRes = await client.get('/projects');
      const proj = projRes.data.find((p) => String(p.id) === String(id));
      if (proj) setProject(proj);
    } catch (err) {
      console.error('Failed to update status:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="text-center py-20">
        <p className="text-slate-400">Project not found.</p>
        <button onClick={() => navigate('/')} className="btn-primary mt-4">
          Back to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Back button */}
      <button
        onClick={() => navigate('/')}
        className="flex items-center gap-2 text-slate-400 hover:text-slate-200 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="text-sm font-medium">Back to Dashboard</span>
      </button>

      {/* Project Header */}
      <div className="card">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-indigo-500/10 rounded-xl">
            <BookOpen className="w-8 h-8 text-indigo-400" />
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-slate-100">{project.title}</h1>
            {project.description && (
              <p className="text-slate-400 mt-1">{project.description}</p>
            )}
          </div>
        </div>

        <div className="mt-6">
          <div className="flex justify-between text-sm text-slate-400 mb-2">
            <span>
              {project.completed_topics} of {project.total_topics} topics completed
            </span>
            <span>{project.percentage}%</span>
          </div>
          <ProgressBar percentage={project.percentage} showLabel={false} size="md" />
        </div>
      </div>

      {/* Topics List */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-200 mb-4">Topics</h2>
        {topics.length === 0 ? (
          <p className="text-slate-400 text-sm">No topics found for this project.</p>
        ) : (
          <div className="divide-y divide-slate-700/30">
            {topics.map((topic) => (
              <TopicItem key={topic.id} topic={topic} onStatusChange={handleStatusChange} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
