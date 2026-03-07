import { useEffect, useRef, useState, useCallback } from 'react'

interface UseWebSocketOptions {
  onMessage?: (data: any) => void
  onConnected?: () => void
  onError?: (error: Event) => void
  onDisconnected?: () => void
  autoReconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

export function useWebSocket(url: string, options: UseWebSocketOptions = {}) {
  const {
    onMessage,
    onConnected,
    onError,
    onDisconnected,
    autoReconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const shouldReconnectRef = useRef(true)
  const reconnectAttemptsRef = useRef(0)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        console.log('[WebSocket] Connected to', url)
        setIsConnected(true)
        reconnectAttemptsRef.current = 0
        onConnected?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage?.(data)
        } catch {
          onMessage?.(event.data)
        }
      }

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
        onError?.(error)
      }

      ws.onclose = () => {
        console.log('[WebSocket] Disconnected')
        setIsConnected(false)
        onDisconnected?.()

        if (autoReconnect && shouldReconnectRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval)
        }
      }

      wsRef.current = ws
    } catch (error) {
      console.error('[WebSocket] Connection failed:', error)
      onError?.(error as Event)
    }
  }, [url, onMessage, onConnected, onError, onDisconnected, autoReconnect, reconnectInterval, maxReconnectAttempts])

  useEffect(() => {
    connect()

    return () => {
      shouldReconnectRef.current = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connect])

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    } else {
      console.warn('[WebSocket] Not connected, cannot send:', data)
    }
  }, [])

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false
    if (wsRef.current) {
      wsRef.current.close()
    }
  }, [])

  return {
    isConnected,
    send,
    disconnect,
    ws: wsRef.current,
  }
}
