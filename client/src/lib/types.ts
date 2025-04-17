import { z } from 'zod';
import { csvRowSchema } from '@shared/schema';

export type NotificationType = 'success' | 'error' | 'warning' | 'info' | null;

export interface Notification {
  type: NotificationType;
  message: string;
}

export interface AuthUser {
  id: number;
  username: string;
}

export interface CsvFileInfo {
  id: number;
  fileName: string;
  fileSize: string;
  status: string;
  uploadedAt: Date;
  userId: number;
}

export type CsvRow = z.infer<typeof csvRowSchema>;

export interface ProcessedCsvData {
  file: CsvFileInfo;
  data: CsvRow[];
}

export interface UploadProgressInfo {
  fileName: string;
  fileSize: string;
  progress: number;
  status: string;
}

export interface APIConfig {
  endpoint: string;
  apiKey: string;
}
