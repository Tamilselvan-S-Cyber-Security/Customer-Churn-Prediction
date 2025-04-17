import { useState, useRef, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Upload, Eye, CircleArrowOutDownLeft } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { useToast } from '@/components/ui/use-toast';
import { formatFileSize, validateCsvFile } from '@/lib/utils';
import { apiRequest } from '@/lib/queryClient';
import { useAuth } from '@/context/AuthContext';
import { CsvFileInfo, UploadProgressInfo } from '@/lib/types';

interface UploadCardProps {
  onUploadSuccess: (fileId: number) => void;
}

export default function UploadCard({ onUploadSuccess }: UploadCardProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgressInfo | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { isAuthenticated } = useAuth();
  
  // Fetch recent uploads
  const { data: recentUploads, isLoading: isLoadingRecent } = useQuery<{files: CsvFileInfo[]}>({
    queryKey: ['/api/csv/recent'],
    enabled: isAuthenticated,
  });

  // Handle file upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (data: FormData) => {
      const response = await apiRequest('POST', '/api/csv/upload', data);
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Success",
        description: "CSV file processed successfully!",
        variant: "success",
      });
      
      // Update upload progress to complete
      if (uploadProgress) {
        setUploadProgress({
          ...uploadProgress,
          progress: 100,
          status: 'Ready to process'
        });
      }
      
      // Invalidate recent uploads query
      queryClient.invalidateQueries({ queryKey: ['/api/csv/recent'] });
      
      // Notify parent component about successful upload
      onUploadSuccess(data.fileId);
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to upload CSV file",
        variant: "destructive",
      });
      
      // Update upload progress to error
      if (uploadProgress) {
        setUploadProgress({
          ...uploadProgress,
          status: 'Failed'
        });
      }
    }
  });
  
  // Handle file drop
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const droppedFile = acceptedFiles[0];
    
    if (droppedFile) {
      const validation = validateCsvFile(droppedFile);
      
      if (!validation.valid) {
        toast({
          title: "Invalid file",
          description: validation.message,
          variant: "destructive",
        });
        return;
      }
      
      setFile(droppedFile);
      
      // Setup upload progress display
      setUploadProgress({
        fileName: droppedFile.name,
        fileSize: formatFileSize(droppedFile.size),
        progress: 0,
        status: 'Ready to upload'
      });
      
      // Simulate upload progress
      simulateUploadProgress();
    }
  }, [toast]);
  
  // Configure dropzone
  const { 
    getRootProps, 
    getInputProps, 
    isDragActive 
  } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    maxSize: 5 * 1024 * 1024, // 5MB
    multiple: false
  });
  
  // Simulate upload progress
  const simulateUploadProgress = () => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += 5;
      
      if (uploadProgress) {
        setUploadProgress({
          ...uploadProgress,
          progress,
          status: progress < 100 ? 'Uploading...' : 'Ready to process'
        });
      }
      
      if (progress >= 100) {
        clearInterval(interval);
      }
    }, 100);
  };
  
  // Process the uploaded file
  const handleProcessFile = async () => {
    if (!file) return;
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    console.log("Uploading file:", file.name);
    
    // Upload the file
    uploadMutation.mutate(formData);
  };
  
  // Clear the form
  const handleClearForm = () => {
    setFile(null);
    setUploadProgress(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // No longer needed as we removed API configuration
  
  // View a previously uploaded file
  const handleViewFile = (fileId: number) => {
    onUploadSuccess(fileId);
  };
  
  // Retry a failed upload
  const handleRetryUpload = async (fileId: number) => {
    // This would typically re-process a failed file
    // For now, we'll just view the file
    onUploadSuccess(fileId);
  };
  
  // Get humanized time since upload
  const getTimeSince = (timestamp: string) => {
    const date = new Date(timestamp);
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
  };
  
  return (
    <>
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold text-neutral-600 mb-4">Upload CSV File</h2>
        <p className="text-neutral-400 text-sm mb-6">
          Upload your CSV files to process and analyze your data. Only .csv files are supported.
        </p>
        
        {!isAuthenticated ? (
          <div className="p-4 border border-orange-200 bg-orange-50 rounded-md">
            <p className="text-orange-600 font-medium">Authentication Required</p>
            <p className="text-orange-500 text-sm mt-1">
              Please log in using the button in the header to upload and process CSV files.
            </p>
          </div>
        ) : (
          <>
            {/* Upload area */}
            <div 
              {...getRootProps()} 
              className={`border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center cursor-pointer hover:border-primary transition-colors ${
                isDragActive ? 'border-primary bg-blue-50' : 'border-neutral-200'
              }`}
            >
              <input {...getInputProps()} ref={fileInputRef} />
              <Upload className="h-10 w-10 text-neutral-300 mb-2" />
              <p className="text-neutral-600 font-medium mb-1">Drag and drop your CSV file here</p>
              <p className="text-neutral-400 text-sm mb-4">or</p>
              <button 
                type="button"
                className="px-4 py-2 bg-primary text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-primary"
                onClick={(e) => {
                  e.stopPropagation(); // Prevent event bubbling
                  if (fileInputRef.current) {
                    fileInputRef.current.click();
                  }
                }}
              >
                Browse Files
              </button>
              <p className="mt-4 text-sm text-neutral-400">Maximum file size: 5MB</p>
            </div>
            
            {/* Upload progress */}
            {uploadProgress && (
              <div className="mt-6">
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium text-neutral-600">{uploadProgress.fileName}</span>
                  <span className="text-sm text-neutral-400">{uploadProgress.fileSize}</span>
                </div>
                <div className="w-full bg-neutral-100 rounded-full h-2.5">
                  <div 
                    className="bg-primary h-2.5 rounded-full" 
                    style={{ width: `${uploadProgress.progress}%` }}
                  ></div>
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-xs text-neutral-400">{uploadProgress.status}</span>
                  <span className="text-xs text-neutral-500">{uploadProgress.progress}%</span>
                </div>
              </div>
            )}
            
            {/* Action buttons */}
            <div className="mt-6 flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3">
              <button 
                type="button"
                className="w-full sm:w-auto flex-1 px-4 py-2 bg-primary text-white rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
                onClick={handleProcessFile}
                disabled={!file || uploadMutation.isPending}
              >
                {uploadMutation.isPending ? 'Processing...' : 'Process File'}
              </button>
              <button 
                type="button"
                className="w-full sm:w-auto px-4 py-2 border border-neutral-300 text-neutral-600 rounded-md font-medium hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-neutral-400"
                onClick={handleClearForm}
                disabled={uploadMutation.isPending}
              >
                Clear
              </button>
            </div>
          </>
        )}
      </div>
      
      {/* Recent uploads - Only show when authenticated */}
      {isAuthenticated && (
        <div className="bg-white shadow rounded-lg p-6 mt-6">
          <h2 className="text-xl font-semibold text-neutral-600 mb-4">Recent Uploads</h2>
          <div className="overflow-hidden">
            {isLoadingRecent ? (
              <div className="py-4 text-center text-neutral-500">Loading recent uploads...</div>
            ) : recentUploads?.files?.length > 0 ? (
              <div>
                {recentUploads.files.map((file: CsvFileInfo) => (
                  <div key={file.id} className="py-3 flex items-center border-b border-neutral-100 last:border-0">
                    <span className={`material-icons mr-3 ${
                      file.status === 'error' ? 'text-red-500' : 'text-neutral-300'
                    }`}>
                      {file.status === 'error' ? 'error' : 'description'}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-neutral-600 truncate">
                        {file.fileName}
                      </p>
                      <p className={`text-xs ${
                        file.status === 'error' ? 'text-red-500' : 'text-neutral-400'
                      }`}>
                        {file.fileSize} • {file.status.charAt(0).toUpperCase() + file.status.slice(1)} • {getTimeSince(file.uploadedAt.toString())}
                      </p>
                    </div>
                    <button 
                      type="button"
                      className="p-1 text-neutral-400 hover:text-primary"
                      onClick={() => file.status === 'error' ? handleRetryUpload(file.id) : handleViewFile(file.id)}
                    >
                      {file.status === 'error' ? (
                        <CircleArrowOutDownLeft className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-4 text-center text-neutral-500">No recent uploads</div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
