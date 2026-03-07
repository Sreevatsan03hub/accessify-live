import { useState } from 'react';
import { Button } from '../ui/Button';

export function RoomCodeDisplay({ code, teacherName, title }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const shareableLink = `${window.location.origin}/room/join?code=${code}`;

  return (
    <div className="text-center py-8">
      <h2 className="text-2xl font-bold mb-4 text-white">{title}</h2>
      <p className="text-gray-400 mb-6">Teacher: {teacherName}</p>

      <div className="bg-primary/20 border-2 border-primary rounded-2xl p-8 mb-6">
        <p className="text-gray-400 text-sm mb-2">Room Code</p>
        <p className="text-5xl font-bold text-primary font-mono mb-4">{code}</p>
        
        <Button
          variant="primary"
          onClick={handleCopy}
          className="mb-4"
        >
          {copied ? '✓ Copied!' : '📋 Copy Code'}
        </Button>
      </div>

      <div className="bg-gray-900 rounded-xl p-4 mb-6">
        <p className="text-gray-400 text-sm mb-2">Share Link</p>
        <p className="text-sm text-gray-300 break-all font-mono">{shareableLink}</p>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => {
            navigator.clipboard.writeText(shareableLink);
            alert('Link copied!');
          }}
          className="mt-2"
        >
          Copy Link
        </Button>
      </div>

      <p className="text-gray-500 text-sm">
        🎯 Students can join by entering this code or clicking the link
      </p>
    </div>
  );
}
