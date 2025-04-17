import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function formatTimeSince(date: Date): string {
  const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
  
  let interval = seconds / 31536000; // seconds in a year
  if (interval > 1) return Math.floor(interval) + ' years ago';
  
  interval = seconds / 2592000; // seconds in a month
  if (interval > 1) return Math.floor(interval) + ' months ago';
  
  interval = seconds / 86400; // seconds in a day
  if (interval > 1) return Math.floor(interval) + ' days ago';
  
  interval = seconds / 3600; // seconds in an hour
  if (interval > 1) return Math.floor(interval) + ' hours ago';
  
  interval = seconds / 60; // seconds in a minute
  if (interval > 1) return Math.floor(interval) + ' minutes ago';
  
  return Math.floor(seconds) + ' seconds ago';
}

export function validateCsvFile(file: File | null): { valid: boolean; message?: string } {
  if (!file) {
    return { valid: false, message: 'No file selected' };
  }
  
  // Check file type
  if (!file.name.toLowerCase().endsWith('.csv')) {
    return { valid: false, message: 'Only CSV files are allowed' };
  }
  
  // Check file size (5MB max)
  const maxSize = 5 * 1024 * 1024; // 5MB
  if (file.size > maxSize) {
    return { valid: false, message: 'File size exceeds the 5MB limit' };
  }
  
  return { valid: true };
}

export function getStatusColor(status: string): {
  bgColor: string;
  textColor: string;
} {
  switch (status.toLowerCase()) {
    case 'active':
      return { bgColor: 'bg-green-100', textColor: 'text-green-800' };
    case 'inactive':
    case 'churned':
      return { bgColor: 'bg-red-100', textColor: 'text-red-800' };
    case 'pending':
      return { bgColor: 'bg-yellow-100', textColor: 'text-yellow-800' };
    case 'error':
      return { bgColor: 'bg-red-100', textColor: 'text-red-800' };
    case 'unknown':
      return { bgColor: 'bg-gray-100', textColor: 'text-gray-800' };
    default:
      return { bgColor: 'bg-blue-100', textColor: 'text-blue-800' };
  }
}
