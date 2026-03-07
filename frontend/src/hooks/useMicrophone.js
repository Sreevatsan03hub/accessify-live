/**
 * useMicrophone — Real microphone capture using AudioWorkletNode
 * Falls back to ScriptProcessorNode if AudioWorklet is unavailable.
 *
 * Captures mic audio → Float32Array → base64 → callback (onAudioChunk)
 */
import { useEffect, useRef, useState, useCallback } from 'react';

// ── Convert Float32Array → base64 ───────────────────────────────────────────
function float32ToBase64(float32Array) {
  const bytes = new Uint8Array(float32Array.buffer);
  let binary = '';
  // Process in chunks to avoid stack overflow on large arrays
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

export function useMicrophone(enabled = false, onAudioChunk = null) {
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [volume, setVolume] = useState(0);

  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const workletNodeRef = useRef(null);   // AudioWorkletNode
  const processorRef = useRef(null);   // ScriptProcessorNode (fallback)
  const analyserRef = useRef(null);
  const animationRef = useRef(null);
  const onAudioChunkRef = useRef(onAudioChunk);

  // Keep ref in sync so the worklet callback always sees the latest handler
  useEffect(() => { onAudioChunkRef.current = onAudioChunk; }, [onAudioChunk]);

  // ── Volume monitor ───────────────────────────────────────────────────────
  const startVolumeMonitor = useCallback(() => {
    if (!analyserRef.current) return;
    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const update = () => {
      analyser.getByteFrequencyData(dataArray);
      const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
      setVolume(Math.round((avg / 255) * 100));
      animationRef.current = requestAnimationFrame(update);
    };
    update();
  }, []);

  // ── Main start ───────────────────────────────────────────────────────────
  const startMicrophone = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // 1. Get microphone stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
          // Note: sampleRate hint is advisory — browser may ignore it
          sampleRate: 16000,
        },
        video: false,
      });
      streamRef.current = stream;

      // 2. AudioContext — request 16kHz
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioCtx;

      // Resume context (required after user gesture in some browsers)
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      const source = audioCtx.createMediaStreamSource(stream);

      // 3. Analyser for volume UI
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      source.connect(analyser);

      // 4a. Try AudioWorklet (Chrome 66+, reliable, not deprecated)
      let usingWorklet = false;
      if (audioCtx.audioWorklet) {
        try {
          await audioCtx.audioWorklet.addModule('/audio-processor.js');
          const workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');
          workletNodeRef.current = workletNode;

          workletNode.port.onmessage = (event) => {
            if (!onAudioChunkRef.current) return;
            const float32 = new Float32Array(event.data.chunk);
            const base64 = float32ToBase64(float32);
            onAudioChunkRef.current(base64, audioCtx.sampleRate);
          };

          source.connect(workletNode);
          // Don't connect to destination — avoids echo
          usingWorklet = true;
          console.log('[Mic] Using AudioWorkletNode ✓');
        } catch (workletErr) {
          console.warn('[Mic] AudioWorklet failed, falling back to ScriptProcessor:', workletErr);
        }
      }

      // 4b. Fallback: ScriptProcessorNode (deprecated but universally supported)
      if (!usingWorklet) {
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (event) => {
          if (!onAudioChunkRef.current) return;
          const inputData = event.inputBuffer.getChannelData(0);
          const chunk = new Float32Array(inputData); // copy
          const base64 = float32ToBase64(chunk);
          onAudioChunkRef.current(base64, audioCtx.sampleRate);
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);
        console.log('[Mic] Using ScriptProcessorNode (fallback)');
      }

      startVolumeMonitor();
      setIsActive(true);
      setIsLoading(false);

    } catch (err) {
      console.error('[Mic] Error:', err);
      setIsLoading(false);
      if (err.name === 'NotAllowedError') setError('Microphone permission denied. Please allow mic access in the browser.');
      else if (err.name === 'NotFoundError') setError('No microphone found. Please connect a microphone.');
      else setError(err.message || 'Failed to access microphone.');
    }
  }, [startVolumeMonitor]);

  // ── Stop ─────────────────────────────────────────────────────────────────
  const stopMicrophone = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => { });
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsActive(false);
    setVolume(0);
  }, []);

  // ── Toggle ───────────────────────────────────────────────────────────────
  const toggleMicrophone = useCallback(async () => {
    if (isActive) stopMicrophone();
    else await startMicrophone();
  }, [isActive, startMicrophone, stopMicrophone]);

  // Auto-start if enabled prop changes
  useEffect(() => {
    if (enabled && !isActive && !isLoading) startMicrophone();
    if (!enabled && isActive) stopMicrophone();
  }, [enabled]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount
  useEffect(() => () => stopMicrophone(), []);  // eslint-disable-line react-hooks/exhaustive-deps

  return { isActive, isLoading, error, volume, startMicrophone, stopMicrophone, toggleMicrophone, stream: streamRef.current };
}
