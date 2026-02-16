import React from 'react';
import { Dialog } from '@headlessui/react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}
export function Modal({
  isOpen,
  onClose,
  title,
  children,
  size = 'md'
}: ModalProps) {
  const sizes = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl'
  };
  return (
    <AnimatePresence>
      {isOpen &&
      <Dialog
        static
        as={motion.div}
        initial={{
          opacity: 0
        }}
        animate={{
          opacity: 1
        }}
        exit={{
          opacity: 0
        }}
        open={isOpen}
        onClose={onClose}
        className="fixed inset-0 z-50 overflow-y-auto">

          <div className="min-h-screen px-4 text-center">
            <Dialog.Overlay className="fixed inset-0 bg-black/30 backdrop-blur-sm" />

            {/* This element is to trick the browser into centering the modal contents. */}
            <span
            className="inline-block h-screen align-middle"
            aria-hidden="true">

              &#8203;
            </span>

            <motion.div
            initial={{
              opacity: 0,
              scale: 0.95,
              y: 20
            }}
            animate={{
              opacity: 1,
              scale: 1,
              y: 0
            }}
            exit={{
              opacity: 0,
              scale: 0.95,
              y: 20
            }}
            className={`inline-block w-full ${sizes[size]} p-6 my-8 overflow-hidden text-left align-middle transition-all transform bg-white shadow-xl rounded-2xl relative`}>

              <button
              onClick={onClose}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-500 focus:outline-none">

                <X className="w-5 h-5" />
              </button>

              {title &&
            <Dialog.Title
              as="h3"
              className="text-lg font-medium leading-6 text-gray-900 mb-4">

                  {title}
                </Dialog.Title>
            }

              <div className="mt-2">{children}</div>
            </motion.div>
          </div>
        </Dialog>
      }
    </AnimatePresence>);

}