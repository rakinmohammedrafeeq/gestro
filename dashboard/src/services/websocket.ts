/**
 * WebSocket Service for Gestro Dashboard
 * Author: Rakin Mohammed Rafeeq
 * Description: Manages WebSocket connections for real-time updates
 */

const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  /**
   * Connect to WebSocket endpoint
   */
  connect(endpoint: string, onMessage: (data: any) => void, onError?: (error: Event) => void) {
    const url = `${WS_BASE_URL}${endpoint}`;
    
    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log(`WebSocket connected: ${endpoint}`);
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) {
          onError(error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.attemptReconnect(endpoint, onMessage, onError);
      };

    } catch (error) {
      console.error('Error creating WebSocket:', error);
    }
  }

  /**
   * Attempt to reconnect to WebSocket
   */
  private attemptReconnect(endpoint: string, onMessage: (data: any) => void, onError?: (error: Event) => void) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Reconnecting... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect(endpoint, onMessage, onError);
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  /**
   * Send data through WebSocket
   */
  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  /**
   * Close WebSocket connection
   */
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export const metricsWsService = new WebSocketService();
export const detectionWsService = new WebSocketService();
