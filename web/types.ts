export interface TelemetryItem<T = any> {
  value: T;
  ts: number;
}

export interface PlantStateData {
  temperature?: TelemetryItem<number>;
  humidity?: TelemetryItem<number>;
  water_low?: TelemetryItem<boolean>;
  pump_on?: TelemetryItem<boolean>;
  manual_mode?: TelemetryItem<boolean>;
}

export interface ApiResponse<T> {
  ok: boolean;
  error?: string;
  data: T;
  detail?: string; // For error cases in the user's snippet
}

export interface ControlState {
  isPending: boolean;
  message: string | null;
  status: 'idle' | 'success' | 'error';
}