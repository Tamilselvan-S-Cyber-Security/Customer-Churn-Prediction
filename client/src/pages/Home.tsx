import { useState } from 'react';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import UploadCard from '@/components/UploadCard';
import ResultsCard from '@/components/ResultsCard';
import NotificationBanner from '@/components/NotificationBanner';
import { Notification } from '@/lib/types';

export default function Home() {
  const [notification, setNotification] = useState<Notification>({
    type: null,
    message: ''
  });
  const [selectedFileId, setSelectedFileId] = useState<number | null>(null);
  
  const showNotification = (type: Notification['type'], message: string) => {
    setNotification({ type, message });
  };
  
  const clearNotification = () => {
    setNotification({ type: null, message: '' });
  };
  
  const handleUploadSuccess = (fileId: number) => {
    setSelectedFileId(fileId);
    showNotification('success', 'CSV file processed successfully!');
  };
  
  return (
    <div className="bg-[#fafafa] min-h-screen flex flex-col">
      <Header title="CSV Data Processor" />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-grow">
        <NotificationBanner
          type={notification.type}
          message={notification.message}
          isVisible={!!notification.type}
          onClose={clearNotification}
        />
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left side - Upload section */}
          <div className="lg:col-span-5">
            <UploadCard onUploadSuccess={handleUploadSuccess} />
          </div>
          
          {/* Right side - Results display */}
          <div className="lg:col-span-7">
            <ResultsCard fileId={selectedFileId} />
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}
