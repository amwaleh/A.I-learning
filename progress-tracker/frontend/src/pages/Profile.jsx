import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { User, Mail, Calendar, Save, LogOut, Loader2 } from 'lucide-react';
import { formatDate } from '../utils/helpers';
import client from '../api/client';

export default function Profile() {
  const { user, logout, updateUser } = useAuth();
  const navigate = useNavigate();
  const [displayName, setDisplayName] = useState(user?.display_name || '');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage('');
    try {
      const { data } = await client.patch('/auth/profile', { display_name: displayName });
      updateUser({ ...user, display_name: displayName, ...data });
      setMessage('Profile updated successfully!');
    } catch (err) {
      setMessage(err.response?.data?.detail || 'Failed to update profile');
    } finally {
      setSaving(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-slate-100">Profile</h1>

      {/* User Info Card */}
      <div className="card space-y-4">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 bg-indigo-500/20 rounded-full flex items-center justify-center">
            <User className="w-8 h-8 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-100">
              {user?.display_name || 'User'}
            </h2>
            <div className="flex items-center gap-2 text-sm text-slate-400 mt-1">
              <Mail className="w-4 h-4" />
              <span>{user?.email}</span>
            </div>
            {user?.created_at && (
              <div className="flex items-center gap-2 text-sm text-slate-400 mt-0.5">
                <Calendar className="w-4 h-4" />
                <span>Joined {formatDate(user.created_at)}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Edit Profile */}
      <form onSubmit={handleSave} className="card space-y-4">
        <h3 className="text-lg font-semibold text-slate-200">Edit Profile</h3>

        {message && (
          <div
            className={`text-sm px-4 py-3 rounded-lg ${
              message.includes('success')
                ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
                : 'bg-rose-500/10 border border-rose-500/30 text-rose-400'
            }`}
          >
            {message}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1.5">Display Name</label>
          <input
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            className="input-field"
            placeholder="Your display name"
          />
        </div>

        <button type="submit" disabled={saving} className="btn-primary flex items-center gap-2">
          {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          Save Changes
        </button>
      </form>

      {/* Logout */}
      <div className="card">
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 text-rose-400 hover:text-rose-300 font-medium transition-colors"
        >
          <LogOut className="w-5 h-5" />
          Sign Out
        </button>
      </div>
    </div>
  );
}
