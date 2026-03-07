import { useState, useRef } from 'react';
import { UploadCloud, FileVideo, X } from 'lucide-react';

const ACCEPTED_FORMATS = ['.mp4', '.mkv', '.avi', '.mov', '.webm'];

export function FileUploader({ onFileSelect, isLoading }) {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => { setIsDragging(false); };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFileChange(f);
  };

  const handleFileChange = (selectedFile) => {
    const ext = '.' + selectedFile.name.split('.').pop().toLowerCase();
    if (!ACCEPTED_FORMATS.includes(ext)) {
      alert(`Unsupported format. Please use: ${ACCEPTED_FORMATS.join(', ')}`);
      return;
    }
    setFile(selectedFile);
    onFileSelect(selectedFile);
  };

  const handleInputChange = (e) => {
    if (e.target.files?.[0]) handleFileChange(e.target.files[0]);
  };

  const clearFile = () => { setFile(null); onFileSelect(null); };

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragging ? '#2563EB' : '#93C5FD'}`,
          borderRadius: '16px',
          padding: '60px 40px',
          background: isDragging ? '#EFF6FF' : '#F8FAFC',
          cursor: isLoading ? 'default' : 'pointer',
          textAlign: 'center',
          transition: 'all 0.2s ease',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_FORMATS.join(',')}
          onChange={handleInputChange}
          className="hidden"
          disabled={isLoading}
        />

        <div className="flex items-center justify-center mb-5">
          <div className="w-16 h-16 rounded-2xl flex items-center justify-center"
            style={{ background: isDragging ? '#DBEAFE' : '#EFF6FF' }}>
            <UploadCloud size={30} style={{ color: '#2563EB' }} />
          </div>
        </div>

        <h3 className="text-base font-bold mb-1" style={{ color: '#0F172A' }}>
          Drag &amp; drop your video here
        </h3>
        <p className="text-sm mb-5" style={{ color: '#64748B' }}>
          or click to browse from your computer
        </p>

        <button
          type="button"
          onClick={e => { e.stopPropagation(); fileInputRef.current?.click(); }}
          disabled={isLoading}
          className="text-sm font-semibold rounded-xl border transition-all disabled:opacity-60"
          style={{
            padding: '10px 24px', background: '#2563EB', color: '#fff',
            border: 'none', boxShadow: '0 4px 12px rgba(37,99,235,0.30)',
            cursor: isLoading ? 'not-allowed' : 'pointer'
          }}
          onMouseEnter={e => { if (!isLoading) e.currentTarget.style.background = '#1D4ED8'; }}
          onMouseLeave={e => { if (!isLoading) e.currentTarget.style.background = '#2563EB'; }}
        >
          Select File
        </button>

        <p className="text-xs mt-4" style={{ color: '#94A3B8' }}>
          Supported: {ACCEPTED_FORMATS.join(', ')}
        </p>
      </div>

      {/* Selected file preview */}
      {file && (
        <div className="mt-4 flex items-center gap-4 rounded-xl p-4"
          style={{ background: '#EFF6FF', border: '1px solid #BFDBFE' }}>
          <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: '#DBEAFE' }}>
            <FileVideo size={20} style={{ color: '#2563EB' }} />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold truncate" style={{ color: '#0F172A' }}>{file.name}</p>
            <p className="text-xs" style={{ color: '#64748B' }}>
              {(file.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>
          {!isLoading && (
            <button onClick={clearFile} className="text-slate-400 hover:text-red-500 transition-colors">
              <X size={18} />
            </button>
          )}
        </div>
      )}
    </div>
  );
}
