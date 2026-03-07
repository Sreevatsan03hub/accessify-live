import api from './api'

export interface CreateRoomResponse {
  room_code: string
  room_id: string
  created_at: string
}

export interface JoinRoomResponse {
  participant_id: string
  websocket_url: string
  room_code: string
}

export async function createRoom(title: string, teacherName: string): Promise<CreateRoomResponse> {
  const response = await api.post('/rooms/create', {
    title,
    teacher_name: teacherName,
  })
  return response.data
}

export async function joinRoom(roomCode: string, name: string, role: string, language: string): Promise<JoinRoomResponse> {
  const response = await api.post(`/rooms/${roomCode}/join`, {
    name,
    role,
    language,
  })
  return response.data
}

export async function leaveRoom(roomCode: string, participantId: string): Promise<void> {
  await api.post(`/rooms/${roomCode}/leave/${participantId}`)
}

export async function getRooms(): Promise<any[]> {
  const response = await api.get('/rooms/')
  return response.data
}

export async function getRoom(roomCode: string): Promise<any> {
  const response = await api.get(`/rooms/${roomCode}`)
  return response.data
}
