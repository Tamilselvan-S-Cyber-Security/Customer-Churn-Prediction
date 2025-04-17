import { X, Check, AlertCircle } from 'lucide-react';
import { NotificationType } from '@/lib/types';
import { cn } from '@/lib/utils';

interface NotificationBannerProps {
  type: NotificationType;
  message: string;
  isVisible: boolean;
  onClose: () => void;
}

export default function NotificationBanner({ 
  type, 
  message, 
  isVisible, 
  onClose 
}: NotificationBannerProps) {
  if (!isVisible || !type) return null;
  
  // Configure notification styles based on type
  const config = {
    success: {
      bg: 'bg-green-50',
      text: 'text-green-800',
      border: 'border-green-200',
      icon: <Check className="h-5 w-5 text-green-600" />
    },
    error: {
      bg: 'bg-red-50',
      text: 'text-red-800',
      border: 'border-red-200',
      icon: <AlertCircle className="h-5 w-5 text-red-600" />
    },
    warning: {
      bg: 'bg-yellow-50',
      text: 'text-yellow-800',
      border: 'border-yellow-200',
      icon: <AlertCircle className="h-5 w-5 text-yellow-600" />
    },
    info: {
      bg: 'bg-blue-50',
      text: 'text-blue-800',
      border: 'border-blue-200',
      icon: <AlertCircle className="h-5 w-5 text-blue-600" />
    }
  };
  
  const { bg, text, border, icon } = config[type];
  
  return (
    <div className="mb-6">
      <div className={cn("rounded-md p-4 border", bg, border)}>
        <div className="flex">
          <div className="flex-shrink-0">
            {icon}
          </div>
          <div className="ml-3">
            <p className={cn("text-sm font-medium", text)}>
              {message}
            </p>
          </div>
          <div className="ml-auto pl-3">
            <div className="-mx-1.5 -my-1.5">
              <button 
                onClick={onClose}
                className={cn("inline-flex rounded-md p-1.5 focus:outline-none focus:ring-2 focus:ring-offset-2", text, `focus:ring-${type === 'error' ? 'red' : type === 'success' ? 'green' : 'blue'}-500`)}
              >
                <span className="sr-only">Dismiss</span>
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
