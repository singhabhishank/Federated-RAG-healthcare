import React, { useCallback, useState, createContext, useContext } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { X, CheckCircle, AlertTriangle, Info, AlertCircle } from 'lucide-react';
type ToastType = 'success' | 'error' | 'warning' | 'info';
interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}
interface ToastContextType {
  addToast: (type: ToastType, message: string, duration?: number) => void;
  removeToast: (id: string) => void;
}
const ToastContext = createContext<ToastContextType | undefined>(undefined);
export function ToastProvider({ children }: {children: React.ReactNode;}) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);
  const addToast = useCallback(
    (type: ToastType, message: string, duration = 5000) => {
      const id = Math.random().toString(36).substr(2, 9);
      setToasts((prev) => [
      ...prev,
      {
        id,
        type,
        message,
        duration
      }]
      );
      if (duration > 0) {
        setTimeout(() => {
          removeToast(id);
        }, duration);
      }
    },
    [removeToast]
  );
  return (
    <ToastContext.Provider
      value={{
        addToast,
        removeToast
      }}>

      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col space-y-2 pointer-events-none">
        <AnimatePresence>
          {toasts.map((toast) =>
          <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
          )}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>);

}
function ToastItem({
  toast,
  onRemove



}: {toast: Toast;onRemove: (id: string) => void;}) {
  const icons = {
    success: <CheckCircle className="w-5 h-5 text-green-500" />,
    error: <AlertCircle className="w-5 h-5 text-red-500" />,
    warning: <AlertTriangle className="w-5 h-5 text-amber-500" />,
    info: <Info className="w-5 h-5 text-blue-500" />
  };
  const styles = {
    success: 'bg-white border-green-100',
    error: 'bg-white border-red-100',
    warning: 'bg-white border-amber-100',
    info: 'bg-white border-blue-100'
  };
  return (
    <motion.div
      initial={{
        opacity: 0,
        y: 20,
        scale: 0.9
      }}
      animate={{
        opacity: 1,
        y: 0,
        scale: 1
      }}
      exit={{
        opacity: 0,
        scale: 0.9,
        transition: {
          duration: 0.2
        }
      }}
      className={`pointer-events-auto flex items-center w-full max-w-sm p-4 rounded-lg shadow-lg border ${styles[toast.type]}`}>

      <div className="flex-shrink-0 mr-3">{icons[toast.type]}</div>
      <div className="flex-1 text-sm font-medium text-gray-900">
        {toast.message}
      </div>
      <button
        onClick={() => onRemove(toast.id)}
        className="ml-4 text-gray-400 hover:text-gray-500 focus:outline-none">

        <X className="w-4 h-4" />
      </button>
    </motion.div>);

}
export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}