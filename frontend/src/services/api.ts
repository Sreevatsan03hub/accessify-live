import axios from 'axios'
import { API_BASE } from '../utils/constants'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('[API Error]:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export default api
