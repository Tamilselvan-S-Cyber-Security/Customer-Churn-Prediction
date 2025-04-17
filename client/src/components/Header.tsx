import { useAuth } from '@/context/AuthContext';
import { useState } from 'react';
import AuthModal from './AuthModal';

interface HeaderProps {
  title: string;
}

export default function Header({ title }: HeaderProps) {
  const { user, isAuthenticated, logout } = useAuth();
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  
  const handleLogout = async () => {
    await logout();
  };
  
  const openAuthModal = () => {
    setIsAuthModalOpen(true);
  };
  
  const closeAuthModal = () => {
    setIsAuthModalOpen(false);
  };
  
  return (
    <>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-semibold text-neutral-600">{title}</h1>
          <div className="flex items-center">
            {isAuthenticated ? (
              <div className="flex items-center">
                <span className="text-sm text-neutral-500 mr-2">Welcome,</span>
                <span className="text-sm font-medium text-neutral-600">{user?.username || "User"}</span>
                <button 
                  onClick={handleLogout}
                  className="ml-4 text-sm text-primary hover:text-blue-700"
                >
                  Logout
                </button>
              </div>
            ) : (
              <button 
                onClick={openAuthModal}
                className="px-4 py-2 text-sm font-medium text-white bg-primary rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-primary"
              >
                Login
              </button>
            )}
          </div>
        </div>
      </header>
      
      <AuthModal isOpen={isAuthModalOpen} onClose={closeAuthModal} />
    </>
  );
}
