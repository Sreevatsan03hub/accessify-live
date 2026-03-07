export const formatTimestamp = (ms) => {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
};

export const parseVTT = (vttContent) => {
  const lines = vttContent.split('\n');
  const captions = [];
  let currentCaption = {};

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (line.includes('-->')) {
      const [start, end] = line.split('-->').map(t => t.trim());
      currentCaption = {
        startTime: vttTimeToMs(start),
        endTime: vttTimeToMs(end),
      };
    } else if (line && !line.startsWith('WEBVTT') && line !== 'NOTE') {
      if (currentCaption.startTime !== undefined) {
        currentCaption.text = line;
        captions.push(currentCaption);
        currentCaption = {};
      }
    }
  }

  return captions;
};

export const vttTimeToMs = (timeStr) => {
  const parts = timeStr.split(':');
  const hours = parseInt(parts[0], 10) || 0;
  const minutes = parseInt(parts[1], 10) || 0;
  const seconds = parseFloat(parts[2]) || 0;

  return (hours * 3600 + minutes * 60 + seconds) * 1000;
};

export const msToVttTime = (ms) => {
  const totalSeconds = ms / 1000;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.000`;
};

export const generateVTT = (captions) => {
  let vtt = 'WEBVTT\n\n';

  captions.forEach((caption, index) => {
    const startTime = msToVttTime(caption.startTime || index * 5000);
    const endTime = msToVttTime(caption.endTime || (index + 1) * 5000);
    vtt += `${startTime} --> ${endTime}\n`;
    vtt += `${caption.text}\n\n`;
  });

  return vtt;
};

export const generateSRT = (captions) => {
  let srt = '';

  captions.forEach((caption, index) => {
    const startTime = msToSrtTime(caption.startTime || index * 5000);
    const endTime = msToSrtTime(caption.endTime || (index + 1) * 5000);
    srt += `${index + 1}\n`;
    srt += `${startTime} --> ${endTime}\n`;
    srt += `${caption.text}\n\n`;
  });

  return srt;
};

export const msToSrtTime = (ms) => {
  const totalSeconds = ms / 1000;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const milliseconds = Math.floor(ms % 1000);

  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')},${String(milliseconds).padStart(3, '0')}`;
};

export const generatePlainText = (captions) => {
  return captions.map(caption => caption.text).join('\n\n');
};

export const generateSummary = (captions) => {
  const text = generatePlainText(captions);
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];

  // Simple summary: take every 3rd sentence or first 5 sentences
  const summaryLength = Math.min(5, Math.ceil(sentences.length / 3));
  const summary = sentences.slice(0, summaryLength).join(' ').trim();

  return summary || text.substring(0, 200);
};

export const highlightKeywords = (text, keywords) => {
  if (!keywords || keywords.length === 0) return text;

  let highlightedText = text;
  keywords.forEach(({ keyword, emoji }) => {
    const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
    highlightedText = highlightedText.replace(
      regex,
      `<span class="keyword-highlight">${emoji} ${keyword}</span>`
    );
  });

  return highlightedText;
};

export const calculateReadingTime = (text) => {
  const wordsPerMinute = 200;
  const wordCount = text.split(/\s+/).length;
  return Math.ceil(wordCount / wordsPerMinute);
};

export const estimateCaptionDuration = (text) => {
  // Rough estimate: ~3-5 words per second
  const words = text.split(/\s+/).length;
  return Math.ceil((words / 3) * 1000); // milliseconds
};
