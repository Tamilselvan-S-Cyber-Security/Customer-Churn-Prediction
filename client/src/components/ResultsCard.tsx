import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { RefreshCw, Download, Maximize, Search, ChevronLeft, ChevronRight } from 'lucide-react';
import { CsvRow } from '@/lib/types';
import { getStatusColor } from '@/lib/utils';
import { useAuth } from '@/context/AuthContext';

interface ResultsCardProps {
  fileId: number | null;
}

export default function ResultsCard({ fileId }: ResultsCardProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<{
    key: keyof CsvRow;
    direction: 'asc' | 'desc';
  } | null>(null);
  
  const rowsPerPage = 5;
  
  // Reset page when fileId changes
  useEffect(() => {
    setCurrentPage(1);
  }, [fileId]);
  
  // Fetch CSV data
  const { 
    data: csvData, 
    isLoading, 
    error,
    refetch
  } = useQuery({
    queryKey: ['/api/csv', fileId].filter(Boolean),
    queryFn: async ({ queryKey }) => {
      try {
        const url = fileId ? `/api/csv/${fileId}` : '/api/csv';
        const response = await fetch(url, {
          credentials: 'include'
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          console.error('Error response:', errorText);
          throw new Error(`Error ${response.status}: ${errorText}`);
        }
        
        return response.json();
      } catch (err) {
        console.error('Query error:', err);
        throw err;
      }
    },
    enabled: !!fileId
  });
  
  const handleRefresh = () => {
    if (fileId) {
      refetch();
    }
  };
  
  const handleDownload = () => {
    if (!csvData?.data) return;
    
    // Create CSV content
    const headers = 'id,name,email,status,lastUpdated\n';
    const content = csvData.data.map((row: CsvRow) => 
      `${row.id},${row.name},${row.email},${row.status},${row.lastUpdated}`
    ).join('\n');
    
    // Create download link
    const blob = new Blob([headers + content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${csvData.file.fileName.replace('.csv', '')}_processed.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  const handleFullscreen = () => {
    const dataDisplay = document.getElementById('data-display');
    if (dataDisplay && document.fullscreenEnabled) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        dataDisplay.requestFullscreen();
      }
    }
  };
  
  const handleSort = (column: keyof CsvRow) => {
    setSortConfig((prevConfig) => {
      if (prevConfig?.key === column) {
        return {
          key: column,
          direction: prevConfig.direction === 'asc' ? 'desc' : 'asc'
        };
      }
      return { key: column, direction: 'asc' };
    });
  };
  
  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    setCurrentPage(1);
  };
  
  const handleStatusFilter = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setStatusFilter(e.target.value);
    setCurrentPage(1);
  };
  
  // Filter and sort data
  const filteredData = csvData?.data ? csvData.data.filter((row: CsvRow) => {
    // Apply search filter
    const matchesSearch = 
      row.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      row.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      row.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
      row.status.toLowerCase().includes(searchTerm.toLowerCase());
    
    // Apply status filter
    const matchesStatus = statusFilter === 'all' || row.status.toLowerCase() === statusFilter.toLowerCase();
    
    return matchesSearch && matchesStatus;
  }) : [];
  
  // Sort data
  const sortedData = sortConfig 
    ? [...filteredData].sort((a: CsvRow, b: CsvRow) => {
        const keyA = a[sortConfig.key];
        const keyB = b[sortConfig.key];
        
        if (keyA < keyB) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (keyA > keyB) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      })
    : filteredData;
    
  // Paginate data
  const totalPages = Math.ceil(sortedData.length / rowsPerPage);
  const paginatedData = sortedData.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage
  );
  
  // Import useAuth from context
  const { isAuthenticated } = useAuth();
  
  // Debug the CSV data
  useEffect(() => {
    if (csvData) {
      console.log('CSV Data received:', csvData);
    }
  }, [csvData]);

  // Determine what to display
  const showEmptyState = !fileId;
  const showLoadingState = isLoading;
  const showErrorState = error !== null;
  const showDataDisplay = !showEmptyState && !showLoadingState && !showErrorState && csvData && csvData.data && csvData.data.length > 0;
  
  return (
    <div className="bg-white shadow rounded-lg p-6 h-full">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-neutral-600">CSV Data Preview</h2>
        <div className="flex space-x-2">
          <button 
            onClick={handleRefresh}
            disabled={!fileId || isLoading}
            className="p-2 text-neutral-400 hover:text-primary rounded-md hover:bg-neutral-50 disabled:opacity-50"
          >
            <RefreshCw className="h-5 w-5" />
          </button>
          <button 
            onClick={handleDownload}
            disabled={!csvData?.data || isLoading}
            className="p-2 text-neutral-400 hover:text-primary rounded-md hover:bg-neutral-50 disabled:opacity-50"
          >
            <Download className="h-5 w-5" />
          </button>
          <button 
            onClick={handleFullscreen}
            disabled={!csvData?.data || isLoading}
            className="p-2 text-neutral-400 hover:text-primary rounded-md hover:bg-neutral-50 disabled:opacity-50"
          >
            <Maximize className="h-5 w-5" />
          </button>
        </div>
      </div>
      
      {/* Empty state */}
      {(showEmptyState || (csvData && (!csvData.data || csvData.data.length === 0))) && (
        <div className="h-96 flex flex-col items-center justify-center">
          <span className="material-icons text-5xl text-neutral-200 mb-4">table_chart</span>
          <h3 className="text-lg font-medium text-neutral-500 mb-2">
            {fileId && csvData ? "No Data Found in CSV" : "No Data to Display"}
          </h3>
          
          {!isAuthenticated ? (
            <div className="text-center">
              <p className="text-sm text-orange-500 font-medium mb-2">Authentication Required</p>
              <p className="text-sm text-neutral-400 max-w-md">
                Please log in using the button in the header to upload, process, and view CSV files.
              </p>
            </div>
          ) : fileId && csvData ? (
            <div className="text-center">
              <p className="text-sm text-neutral-500 mb-2">The CSV file was processed, but no valid data rows were found.</p>
              <p className="text-sm text-neutral-400 max-w-md">
                Make sure your CSV contains data with the expected columns (ID, Name, Email, Status, Last Updated).
              </p>
            </div>
          ) : (
            <p className="text-sm text-neutral-400 text-center max-w-md">
              Upload a CSV file to see the processed data preview here. Your data will be securely processed and displayed in a structured format.
            </p>
          )}
        </div>
      )}
      
      {/* Loading state */}
      {showLoadingState && (
        <div className="h-96 flex flex-col items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
          <h3 className="text-lg font-medium text-neutral-500 mb-2">Processing Data</h3>
          <p className="text-sm text-neutral-400">Please wait while we process your CSV file...</p>
        </div>
      )}
      
      {/* Error state */}
      {showErrorState && (
        <div className="h-96 flex flex-col items-center justify-center">
          <span className="material-icons text-5xl text-red-500 mb-4">error_outline</span>
          <h3 className="text-lg font-medium text-red-500 mb-2">Error Processing CSV</h3>
          <p className="text-sm text-neutral-500 text-center max-w-md">
            {error instanceof Error ? error.message : 'An error occurred while processing your CSV file.'}
          </p>
          <button 
            onClick={handleRefresh}
            className="mt-4 px-4 py-2 bg-primary text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-primary"
          >
            Try Again
          </button>
        </div>
      )}
      
      {/* Data display */}
      {showDataDisplay && (
        <div id="data-display" className="h-[calc(100%-2rem)]">
          {/* Search and filter */}
          <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3 mb-4">
            <div className="relative flex-1">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-neutral-400" />
              </span>
              <input 
                type="text" 
                placeholder="Search data..." 
                className="block w-full pl-10 pr-3 py-2 border border-neutral-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
                value={searchTerm}
                onChange={handleSearch}
              />
            </div>
            <select 
              className="form-select w-full sm:w-auto px-3 py-2 border border-neutral-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
              value={statusFilter}
              onChange={handleStatusFilter}
            >
              <option value="all">All Statuses</option>
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
              <option value="pending">Pending</option>
              <option value="error">Error</option>
            </select>
          </div>
          
          {/* Table */}
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-neutral-200">
              <thead className="bg-neutral-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    <div 
                      className="flex items-center cursor-pointer" 
                      onClick={() => handleSort('id')}
                    >
                      ID
                      <span className="material-icons ml-1 text-neutral-400 text-sm">
                        {sortConfig?.key === 'id' 
                          ? sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward' 
                          : 'unfold_more'}
                      </span>
                    </div>
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    <div 
                      className="flex items-center cursor-pointer" 
                      onClick={() => handleSort('name')}
                    >
                      Name
                      <span className="material-icons ml-1 text-neutral-400 text-sm">
                        {sortConfig?.key === 'name' 
                          ? sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward' 
                          : 'unfold_more'}
                      </span>
                    </div>
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    <div 
                      className="flex items-center cursor-pointer" 
                      onClick={() => handleSort('email')}
                    >
                      Email
                      <span className="material-icons ml-1 text-neutral-400 text-sm">
                        {sortConfig?.key === 'email' 
                          ? sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward' 
                          : 'unfold_more'}
                      </span>
                    </div>
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    <div 
                      className="flex items-center cursor-pointer" 
                      onClick={() => handleSort('status')}
                    >
                      Status
                      <span className="material-icons ml-1 text-neutral-400 text-sm">
                        {sortConfig?.key === 'status' 
                          ? sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward' 
                          : 'unfold_more'}
                      </span>
                    </div>
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    <div 
                      className="flex items-center cursor-pointer" 
                      onClick={() => handleSort('lastUpdated')}
                    >
                      Last Updated
                      <span className="material-icons ml-1 text-neutral-400 text-sm">
                        {sortConfig?.key === 'lastUpdated' 
                          ? sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward' 
                          : 'unfold_more'}
                      </span>
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-neutral-100">
                {paginatedData.length > 0 ? (
                  paginatedData.map((row: CsvRow, index: number) => {
                    const { bgColor, textColor } = getStatusColor(row.status);
                    return (
                      <tr key={index} className="hover:bg-neutral-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-neutral-600">
                          {row.id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-500">
                          {row.name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-500">
                          {row.email}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${bgColor} ${textColor}`}>
                            {row.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-500">
                          {row.lastUpdated}
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={5} className="px-6 py-4 text-center text-sm text-neutral-500">
                      No data matching your filters
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          
          {/* Pagination */}
          {sortedData.length > 0 && (
            <div className="flex items-center justify-between border-t border-neutral-200 bg-white px-4 py-3 sm:px-6 mt-4">
              <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm text-neutral-500">
                    Showing <span className="font-medium">{((currentPage - 1) * rowsPerPage) + 1}</span> to{' '}
                    <span className="font-medium">
                      {Math.min(currentPage * rowsPerPage, sortedData.length)}
                    </span> of{' '}
                    <span className="font-medium">{sortedData.length}</span> results
                  </p>
                </div>
                <div>
                  <nav className="isolate inline-flex -space-x-px rounded-md shadow-sm" aria-label="Pagination">
                    <button
                      onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                      disabled={currentPage === 1}
                      className="relative inline-flex items-center rounded-l-md px-2 py-2 text-neutral-400 ring-1 ring-inset ring-neutral-300 hover:bg-neutral-50 focus:z-20 focus:outline-offset-0 disabled:opacity-50"
                    >
                      <span className="sr-only">Previous</span>
                      <ChevronLeft className="h-5 w-5" />
                    </button>
                    
                    {Array.from({ length: Math.min(3, totalPages) }, (_, i) => {
                      // Show current page and pages around it
                      let pageNum;
                      
                      if (totalPages <= 3) {
                        // If there are 3 or fewer pages, show all
                        pageNum = i + 1;
                      } else if (currentPage <= 2) {
                        // If we're near the start, show first 3 pages
                        pageNum = i + 1;
                      } else if (currentPage >= totalPages - 1) {
                        // If we're near the end, show last 3 pages
                        pageNum = totalPages - 2 + i;
                      } else {
                        // Otherwise show current page and surrounding pages
                        pageNum = currentPage - 1 + i;
                      }
                      
                      const isActive = pageNum === currentPage;
                      
                      return (
                        <button
                          key={pageNum}
                          onClick={() => setCurrentPage(pageNum)}
                          className={`relative inline-flex items-center px-4 py-2 text-sm font-semibold ${
                            isActive 
                              ? 'bg-primary z-10 text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary'
                              : 'text-neutral-600 ring-1 ring-inset ring-neutral-300 hover:bg-neutral-50 focus:z-20 focus:outline-offset-0'
                          }`}
                        >
                          {pageNum}
                        </button>
                      );
                    })}
                    
                    <button
                      onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                      disabled={currentPage === totalPages}
                      className="relative inline-flex items-center rounded-r-md px-2 py-2 text-neutral-400 ring-1 ring-inset ring-neutral-300 hover:bg-neutral-50 focus:z-20 focus:outline-offset-0 disabled:opacity-50"
                    >
                      <span className="sr-only">Next</span>
                      <ChevronRight className="h-5 w-5" />
                    </button>
                  </nav>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
