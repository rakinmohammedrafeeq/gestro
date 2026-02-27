/**
 * API Service for Gestro Dashboard
 * Author: Rakin Mohammed Rafeeq
 * Description: Handles API communication with backend
 */

import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Health check
   */
  async healthCheck() {
    const response = await this.client.get('/health');
    return response.data;
  }

  /**
   * Get system metrics
   */
  async getMetrics() {
    const response = await this.client.get('/metrics');
    return response.data;
  }

  /**
   * Get model information
   */
  async getModelInfo() {
    const response = await this.client.get('/model/info');
    return response.data;
  }

  /**
   * Upload image for detection
   */
  async detectImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post('/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  /**
   * Detect from base64 image
   */
  async detectFromBase64(imageData: string) {
    const response = await this.client.post('/api/v1/detection/base64', {
      image: imageData,
    });

    return response.data;
  }
}

export const apiService = new ApiService();
export default apiService;
