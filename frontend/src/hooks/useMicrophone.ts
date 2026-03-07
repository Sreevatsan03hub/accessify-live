import { useEffect, useRef, useState, useCallback } from 'react'
import { float32ToBase64 } from '../utils/audioUtils'

interface UseMicrophoneOptions {
  onAudioData?: (base64: string) => void
  onError?: (error: Error) => void
  sampleRate?: number
  bufferSize?: number
}

export function useMicrophone({
  onAudioData,
  onError,
  sampleRate = 16000,
  bufferSize = 4096,
}: UseMicrophoneOptions = {}) {
  const [isRecording, setIsRecording] = useState(false)
  const [isMicEnabled, setIsMicEnabled] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaAudioSourceNode | null>(null)

  const startRecording = useCallback(async () => {
    try {
      setError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
        },
      })

      streamRef.current = stream
      setIsMicEnabled(true)

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate,
      })
      audioContextRef.current = audioContext

      const source = audioContext.createMediaStreamSource(stream)
      sourceRef.current = source

      const scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1)
      scriptProcessorRef.current = scriptProcessor

      scriptProcessor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0)
        const float32Array = new Float32Array(inputData)
        const base64 = float32ToBase64(float32Array)
        onAudioData?.(base64)
      }

      source.connect(scriptProcessor)
      scriptProcessor.connect(audioContext.destination)

      setIsRecording(true)
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      setError(error.message)
      console.error('[Microphone] Error starting recording:', error)
      onError?.(error)
      setIsMicEnabled(false)
    }
  }, [onAudioData, onError, sampleRate, bufferSize])

  const stopRecording = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect()
    }

    if (sourceRef.current) {
      sourceRef.current.disconnect()
    }

    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    setIsRecording(false)
    setIsMicEnabled(false)
  }, [])

  const toggleRecording = useCallback(async () => {
    if (isRecording) {
      stopRecording()
    } else {
      await startRecording()
    }
  }, [isRecording, startRecording, stopRecording])

  useEffect(() => {
    return () => {
      stopRecording()
    }
  }, [stopRecording])

  return {
    isRecording,
    isMicEnabled,
    error,
    startRecording,
    stopRecording,
    toggleRecording,
  }
}
