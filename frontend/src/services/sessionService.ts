import api from './api'

export interface Session {
  id: string
  title: string
  type: 'video' | 'live'
  created_at: string
  caption_count: number
  duration?: number
}

export async function getSessions(): Promise<Session[]> {
  const response = await api.get('/sessions/')
  return response.data
}

export async function getSession(id: string): Promise<Session> {
  const response = await api.get(`/sessions/${id}`)
  return response.data
}

export async function deleteSession(id: string): Promise<void> {
  await api.delete(`/sessions/${id}`)
}
