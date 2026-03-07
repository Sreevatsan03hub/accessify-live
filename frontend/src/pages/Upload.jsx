import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { FileUploader } from '../components/upload/FileUploader';
import { UploadCloud, Zap, Target, FileText } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const LANG_OPTIONS = [
  { value: 'none', label: 'None (English only)' },
  { value: 'hi', label: 'हिंदी (Hindi)' },
  { value: 'ta', label: 'தமிழ் (Tamil)' },
  { value: 'te', label: 'తెలుగు (Telugu)' },
];

export function Upload() {
  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState('en');
  const [translateTo, setTranslateTo] = useState('none');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setUploadProgress(0);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    try {
      setIsLoading(true);
      setUploadProgress(0);
      setError(null);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('language', language);
      if (translateTo !== 'none') formData.append('translate_to', translateTo);

      const response = await axios.post(`${API_BASE}/api/v1/video/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(Math.min(percent * 0.5, 50));
        },
        timeout: 900000, // 15 min timeout for large videos
      });

      setUploadProgress(100);
      setTimeout(() => navigate('/player', { state: { videoData: response.data } }), 500);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || err.message || 'Upload failed. Please try again.');
      setIsLoading(false);
      setUploadProgress(0);
    }
  };

  const progressLabel = uploadProgress < 50 ? 'Uploading video…' : 'AI processing captions…';

  return (
    <div className="min-h-screen py-12" style={{ background: '#F1F5F9' }}>
      <div style={{ maxWidth: '860px', margin: '0 auto', padding: '0 24px' }}>

        {/* Page header */}
        <div className="mb-8">
          <h1 className="text-[34px] font-bold mb-1" style={{ color: '#0F172A', letterSpacing: '-0.02em' }}>
            Upload Video
          </h1>
          <p style={{ color: '#64748B', fontSize: '15px' }}>
            Upload a video to automatically generate AI-powered captions and transcripts.
          </p>
        </div>

        {/* Main card */}
        <div className="rounded-2xl p-8 mb-6" style={{
          background: '#fff', border: '1px solid #E2E8F0',
          boxShadow: '0 8px 24px rgba(0,0,0,0.05)'
        }}>

          {/* Step 1 */}
          <div className="mb-8">
            <p className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: '#2563EB' }}>Step 1</p>
            <h2 className="text-lg font-bold mb-4" style={{ color: '#0F172A' }}>Select Video File</h2>
            <FileUploader onFileSelect={handleFileSelect} isLoading={isLoading} />
          </div>

          <hr style={{ border: 'none', borderTop: '1px solid #E2E8F0', margin: '0 -32px 32px' }} />

          {/* Step 2 */}
          <div className="mb-8">
            <p className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: '#2563EB' }}>Step 2</p>
            <h2 className="text-lg font-bold mb-5" style={{ color: '#0F172A' }}>Language Settings</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div>
                <label className="block text-sm font-semibold mb-1.5" style={{ color: '#0F172A' }}>
                  Source Language
                </label>
                <div className="flex items-center gap-2 rounded-xl px-4 py-3"
                  style={{ background: '#F8FAFC', border: '1.5px solid #E2E8F0' }}>
                  <span>🇬🇧</span>
                  <span style={{ color: '#64748B', fontSize: '14px' }}>English (currently supported)</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold mb-1.5" style={{ color: '#0F172A' }}>
                  Translate captions to:
                </label>
                <select
                  value={translateTo}
                  onChange={(e) => setTranslateTo(e.target.value)}
                  className="input-field"
                  disabled={isLoading}
                >
                  {LANG_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="mb-6 rounded-xl p-4 flex items-start gap-3"
              style={{ background: '#FEF2F2', border: '1px solid #FECACA' }}>
              <span style={{ color: '#DC2626', fontSize: '14px' }}>⚠️ {error}</span>
            </div>
          )}

          {/* Step 3 — Upload */}
          {file && (
            <>
              <hr style={{ border: 'none', borderTop: '1px solid #E2E8F0', margin: '0 -32px 32px' }} />
              <div>
                <p className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: '#2563EB' }}>Step 3</p>
                <h2 className="text-lg font-bold mb-5" style={{ color: '#0F172A' }}>Upload &amp; Process</h2>

                {isLoading ? (
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span style={{ color: '#64748B', fontSize: '14px' }}>{progressLabel}</span>
                      <span className="text-sm font-bold" style={{ color: '#0F172A' }}>{Math.round(uploadProgress)}%</span>
                    </div>
                    <div className="w-full rounded-full h-2.5 mb-4" style={{ background: '#E2E8F0' }}>
                      <div
                        className="h-2.5 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%`, background: 'linear-gradient(90deg,#2563EB,#60A5FA)' }}
                      />
                    </div>
                    <div className="text-sm space-y-1" style={{ color: '#64748B' }}>
                      {uploadProgress > 5 && <p>✓ Uploading video file…</p>}
                      {uploadProgress > 50 && <p>✓ Extracting audio…</p>}
                      {uploadProgress > 65 && <p>✓ Running speech-to-text (Whisper)…</p>}
                      {uploadProgress > 80 && <p>✓ AI enrichment &amp; keyword detection…</p>}
                      {uploadProgress > 90 && <p>✓ Generating captions &amp; transcripts…</p>}
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={handleUpload}
                    className="w-full flex items-center justify-center gap-2 py-3.5 rounded-xl font-semibold text-white transition-all"
                    style={{
                      background: '#2563EB', boxShadow: '0 8px 20px rgba(37,99,235,0.30)',
                      fontSize: '15px'
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = '#1D4ED8'}
                    onMouseLeave={e => e.currentTarget.style.background = '#2563EB'}
                  >
                    <UploadCloud size={18} />
                    Upload &amp; Process Video
                  </button>
                )}
              </div>
            </>
          )}
        </div>

        {/* Info strip */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { Icon: Zap, color: '#2563EB', bg: '#DBEAFE', title: 'Fast Processing', desc: 'Most videos process in under 5 minutes' },
            { Icon: Target, color: '#059669', bg: '#D1FAE5', title: 'Accurate', desc: 'Whisper AI speech recognition, high accuracy' },
            { Icon: FileText, color: '#D97706', bg: '#FEF3C7', title: 'Multiple Formats', desc: 'Download as SRT, VTT, or plain text' },
          ].map(({ Icon, color, bg, title, desc }) => (
            <div key={title} className="rounded-xl p-4 flex items-start gap-3"
              style={{ background: '#fff', border: '1px solid #E2E8F0' }}>
              <div className="w-9 h-9 rounded-lg flex-shrink-0 flex items-center justify-center"
                style={{ background: bg }}>
                <Icon size={17} style={{ color }} />
              </div>
              <div>
                <p className="text-sm font-bold" style={{ color: '#0F172A' }}>{title}</p>
                <p className="text-xs mt-0.5" style={{ color: '#64748B' }}>{desc}</p>
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}
