import api from './api'

export interface UploadResponse {
  success: boolean
  filename: string
  duration: number
  transcription: {
    text: string
    language: string
    processing_time: number
    vtt: string
  }
  enrichment: {
    keywords: Array<{ keyword: string; emoji: string; score: number }>
  }
  tone: {
    emotion: string
    intent: string
    emoji: string
  }
  translation?: {
    text: string
    target_language: string
  }
  session_id: string
}

export async function uploadVideo(
  file: File,
  language: string = 'en',
  translateTo?: string,
  onProgress?: (progress: number) => void
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('language', language)
  if (translateTo) {
    formData.append('translate_to', translateTo)
  }

  const response = await api.post('/video/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total) {
        const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100)
        onProgress?.(progress)
      }
    },
  })

  return response.data
}
