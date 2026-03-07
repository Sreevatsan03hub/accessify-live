/**
 * useWebSocket — Real WebSocket connection with auto-reconnect
 * Connects to backend ws://localhost:8001/ws/...
 */
import { useEffect, useRef, useState, useCallback } from 'react';

const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8001';

export function useWebSocket(path, onMessage, enabled = true) {
  const [isConnected, setIsConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const onMessageRef = useRef(onMessage);
  const maxReconnectAttempts = 10;
  const pingIntervalRef = useRef(null);

  // Always keep onMessage ref current (safe to assign during render)
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (!enabled || !path) return;

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const url = path.startsWith('ws') ? path : `${WS_BASE}${path}`;
    console.log('[WS] Connecting to:', url);

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected:', url);
      setIsConnected(true);
      setIsReconnecting(false);
      reconnectAttemptsRef.current = 0;

      // Send ping every 30 seconds to keep connection alive
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'pong') return; // Ignore pong responses
        if (onMessageRef.current) {
          onMessageRef.current(data);
        }
      } catch (e) {
        console.warn('[WS] Failed to parse message:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('[WS] Error:', error);
    };

    ws.onclose = (event) => {
      console.log('[WS] Disconnected. Code:', event.code);
      setIsConnected(false);

      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }

      // Auto-reconnect unless intentionally closed (code 1000)
      if (event.code !== 1000 && enabled && reconnectAttemptsRef.current < maxReconnectAttempts) {
        setIsReconnecting(true);
        reconnectAttemptsRef.current += 1;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 15000);
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);
        reconnectTimeoutRef.current = setTimeout(connect, delay);
      } else {
        setIsReconnecting(false);
      }
    };
  }, [enabled, path]);

  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
      }
    };
  }, [enabled, path]);

  const send = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
      return true;
    }
    console.warn('[WS] Cannot send — not connected');
    return false;
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
    reconnectAttemptsRef.current = maxReconnectAttempts; // Prevent auto-reconnect
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
    }
    setIsConnected(false);
    setIsReconnecting(false);
  }, []);

  return {
    isConnected,
    isReconnecting,
    send,
    disconnect,
    reconnect: connect,
  };
}
