// Audio utilities for microphone capture and WebRTC
// Note: These are stubbed for UI development with mock data

export const float32ArrayToBase64 = (array) => {
  // Stub: In production, convert Float32Array to binary and then base64
  const buffer = new Uint8Array(array.buffer);
  let binary = '';
  for (let i = 0; i < buffer.length; i++) {
    binary += String.fromCharCode(buffer[i]);
  }
  return btoa(binary);
};

export const base64ToFloat32Array = (base64String) => {
  // Stub: In production, decode base64 to binary and then to Float32Array
  const binary = atob(base64String);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
};

export const downloadAudioBlob = (blob, filename = 'audio.wav') => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

export const downloadTextFile = (content, filename = 'content.txt') => {
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

export const downloadJSON = (data, filename = 'data.json') => {
  const json = JSON.stringify(data, null, 2);
  downloadTextFile(json, filename);
};

export const calculateAudioDuration = (audioBuffer) => {
  if (!audioBuffer) return 0;
  return (audioBuffer.length / audioBuffer.sampleRate) * 1000; // milliseconds
};

export const getFrequencyData = (analyser) => {
  const dataArray = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(dataArray);
  return dataArray;
};

export const getWaveformData = (analyser) => {
  const dataArray = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(dataArray);
  return dataArray;
};

// Stub: Mock microphone API
export const mockGetUserMedia = async (constraints) => {
  console.log('[v0] Mock getUserMedia called with constraints:', constraints);
  // In production, would return: navigator.mediaDevices.getUserMedia(constraints)
  // For mock, we return a dummy stream
  return new Promise((resolve) => {
    setTimeout(() => {
      // Return a mock MediaStream with video/audio tracks
      const mockStream = new MediaStream();
      resolve(mockStream);
    }, 100);
  });
};

export const mockCreateAudioContext = () => {
  console.log('[v0] Mock AudioContext created');
  // Return a mock AudioContext (not a real one)
  return {
    createMediaStreamSource: () => ({}),
    createScriptProcessor: () => ({}),
    sampleRate: 16000,
  };
};
