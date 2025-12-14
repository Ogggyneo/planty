import { ApiResponse, PlantStateData } from '../types';

export const fetchTelemetry = async (): Promise<ApiResponse<PlantStateData>> => {
  try {
    const response = await fetch('/api/state');
    const json = await response.json();
    return json;
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : 'Network error',
      data: {} as PlantStateData
    };
  }
};

export const setManualMode = async (enabled: boolean): Promise<ApiResponse<any>> => {
  try {
    const response = await fetch('/api/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled })
    });
    const json = await response.json();
    if (!response.ok) {
      return { ok: false, error: json.detail || 'Request failed', data: null };
    }
    return { ok: true, data: json };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Network error', data: null };
  }
};

export const setPumpState = async (state: boolean): Promise<ApiResponse<any>> => {
  try {
    const response = await fetch('/api/pump', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state })
    });
    const json = await response.json();
    if (!response.ok) {
      return { ok: false, error: json.detail || 'Request failed', data: null };
    }
    return { ok: true, data: json };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Network error', data: null };
  }
};