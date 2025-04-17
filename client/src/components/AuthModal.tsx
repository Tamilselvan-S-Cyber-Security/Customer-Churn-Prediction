import { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Eye, EyeOff } from 'lucide-react';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AuthModal({ isOpen, onClose }: AuthModalProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { login, isLoading } = useAuth();
  
  if (!isOpen) return null;
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const success = await login(username, password);
    if (success) {
      onClose();
      // Reset form
      setUsername('');
      setPassword('');
    }
  };
  
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg max-w-md w-full mx-4 overflow-hidden shadow-xl transform transition-all">
        <div className="px-6 pt-6 pb-4">
          <h3 className="text-lg font-medium text-neutral-600 mb-4">Authentication Required</h3>
          <p className="text-sm text-neutral-500 mb-6">
            Please enter your credentials to access this resource.
          </p>
          
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="auth-email" className="block text-sm text-neutral-500 mb-1">Email</label>
              <input 
                type="email" 
                id="auth-email" 
                className="w-full px-3 py-2 border border-neutral-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary" 
                placeholder="Enter your email"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
            <div>
              <label htmlFor="auth-password" className="block text-sm text-neutral-500 mb-1">Password</label>
              <div className="relative">
                <input 
                  type={showPassword ? "text" : "password"}
                  id="auth-password" 
                  className="w-full px-3 py-2 border border-neutral-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary" 
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button 
                  type="button"
                  className="absolute inset-y-0 right-0 px-3 flex items-center" 
                  onClick={togglePasswordVisibility}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4 text-neutral-400" />
                  ) : (
                    <Eye className="h-4 w-4 text-neutral-400" />
                  )}
                </button>
              </div>
            </div>
            <div className="flex items-center">
              <input 
                type="checkbox" 
                id="remember-me" 
                className="h-4 w-4 text-primary focus:ring-primary border-neutral-300 rounded"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              <label htmlFor="remember-me" className="ml-2 block text-sm text-neutral-500">
                Remember me
              </label>
            </div>
          </form>
        </div>
        
        <div className="bg-neutral-50 px-6 py-4 flex justify-end space-x-3">
          <button 
            className="px-4 py-2 border border-neutral-300 text-neutral-600 rounded-md font-medium hover:bg-neutral-100 focus:outline-none focus:ring-2 focus:ring-neutral-400"
            onClick={onClose}
            disabled={isLoading}
          >
            Cancel
          </button>
          <button 
            className="px-4 py-2 bg-primary text-white rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
            onClick={handleSubmit}
            disabled={isLoading}
          >
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </div>
      </div>
    </div>
  );
}
