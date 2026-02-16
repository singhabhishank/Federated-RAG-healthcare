import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
interface CardProps extends HTMLMotionProps<'div'> {
  children: React.ReactNode;
  variant?: 'default' | 'glass' | 'outlined';
  padding?: 'none' | 'sm' | 'md' | 'lg';
}
export function Card({
  children,
  variant = 'default',
  padding = 'md',
  className = '',
  ...props
}: CardProps) {
  const variants = {
    default: 'bg-white shadow-sm border border-gray-100',
    glass: 'bg-white/80 backdrop-blur-md border border-white/20 shadow-lg',
    outlined: 'bg-transparent border border-gray-200'
  };
  const paddings = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  };
  return (
    <motion.div
      className={`rounded-xl ${variants[variant]} ${paddings[padding]} ${className}`}
      initial={{
        opacity: 0,
        y: 10
      }}
      animate={{
        opacity: 1,
        y: 0
      }}
      transition={{
        duration: 0.3
      }}
      {...props}>

      {children}
    </motion.div>);

}