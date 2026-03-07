import api from './api'

export async function downloadFile(sessionId: string, format: 'srt' | 'vtt' | 'txt' | 'summary'): Promise<Blob> {
  const response = await api.get(`/export/${sessionId}/${format}`, {
    responseType: 'blob',
  })
  return response.data
}

export function triggerDownload(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}

export async function downloadCaption(sessionId: string, format: 'srt' | 'vtt' | 'txt' | 'summary'): Promise<void> {
  try {
    const blob = await downloadFile(sessionId, format)
    const timestamp = new Date().toISOString().split('T')[0]
    const filename = `captions_${timestamp}.${format}`
    triggerDownload(blob, filename)
  } catch (error) {
    console.error('Failed to download caption:', error)
    throw error
  }
}
