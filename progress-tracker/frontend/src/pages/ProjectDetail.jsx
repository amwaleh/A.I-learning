import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import client from '../api/client';
import TopicItem from '../components/TopicItem';
import TopicContent from '../components/TopicContent';
import ProgressBar from '../components/ProgressBar';
import { ArrowLeft, Loader2, BookOpen, PanelLeftClose, PanelLeftOpen } from 'lucide-react';

export default function ProjectDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState(null);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const fetchData = async () => {
    try {
      const [projRes, topicsRes] = await Promise.all([
        client.get('/projects'),
        client.get(`/projects/${id}/topics`),
      ]);
      const proj = projRes.data.find((p) => String(p.id) === String(id));
      setProject(proj);
      setTopics(topicsRes.data.topics || []);
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
      const updateTopicStatus = (items) =>
        items.map((t) => {
          if (t.id === topicId) return { ...t, status: newStatus };
          if (t.children?.length) return { ...t, children: updateTopicStatus(t.children) };
          return t;
        });
      setTopics(updateTopicStatus(topics));

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
    <div className="space-y-4">
      {/* Project Header Bar */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="text-sm font-medium">Back</span>
        </button>
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div className="p-2 bg-indigo-500/10 rounded-lg">
            <BookOpen className="w-5 h-5 text-indigo-400" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="text-lg font-bold text-slate-100 truncate">{project.title}</h1>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-slate-400">
          <span className="hidden sm:inline">
            {project.completed_topics}/{project.total_topics} topics
          </span>
          <div className="w-32 hidden md:block">
            <ProgressBar percentage={project.progress_percentage} showLabel={false} size="sm" />
          </div>
          <span className="font-medium text-indigo-400">{project.progress_percentage}%</span>
        </div>
      </div>

      {/* Main layout: sidebar + content */}
      <div className="flex gap-4" style={{ height: 'calc(100vh - 10rem)' }}>
        {/* Left Sidebar – Topic Navigation */}
        <aside
          className={`flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden
                      ${sidebarCollapsed ? 'w-12' : 'w-80'}`}
        >
          <div className="h-full bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col">
            {/* Sidebar Header */}
            <div className="flex items-center justify-between px-3 py-3 border-b border-slate-800">
              {!sidebarCollapsed && (
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Topics</h2>
              )}
              <button
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                {sidebarCollapsed ? (
                  <PanelLeftOpen className="w-4 h-4" />
                ) : (
                  <PanelLeftClose className="w-4 h-4" />
                )}
              </button>
            </div>

            {/* Topic Tree */}
            {!sidebarCollapsed && (
              <div className="flex-1 overflow-y-auto overflow-x-hidden px-1 py-2 scrollbar-thin">
                {topics.length === 0 ? (
                  <p className="text-slate-500 text-sm px-3 py-4">No topics found.</p>
                ) : (
                  topics.map((topic) => (
                    <TopicItem
                      key={topic.id}
                      topic={topic}
                      onStatusChange={handleStatusChange}
                      onSelect={(t) => setSelectedTopic(t)}
                      selectedTopicId={selectedTopic?.id}
                      compact
                    />
                  ))
                )}
              </div>
            )}
          </div>
        </aside>

        {/* Right Content Area */}
        <main className="flex-1 min-w-0 overflow-hidden">
          {selectedTopic ? (
            <TopicContent
              topicId={selectedTopic.id}
              topicTitle={selectedTopic.title}
              onClose={() => setSelectedTopic(null)}
              fullHeight
            />
          ) : (
            <div className="h-full bg-slate-900/30 border border-slate-800/50 rounded-xl flex flex-col items-center justify-center text-center p-8">
              <BookOpen className="w-16 h-16 text-slate-700 mb-4" />
              <h3 className="text-lg font-medium text-slate-400 mb-2">Select a topic to begin</h3>
              <p className="text-sm text-slate-500 max-w-md">
                Choose a topic from the sidebar to view its learning content.
                Click the status icon to track your progress.
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
