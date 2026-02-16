import React from 'react';
import { motion } from 'framer-motion';
interface GaugeProps {
  value: number; // 0 to 1
  label?: string;
  size?: number;
  showValue?: boolean;
}
export function Gauge({
  value,
  label,
  size = 120,
  showValue = true
}: GaugeProps) {
  // Clamp value between 0 and 1
  const clampedValue = Math.min(Math.max(value, 0), 1);
  // Calculate color based on value (teal -> amber -> red)
  const getColor = (v: number) => {
    if (v < 0.5) return '#0D9488'; // teal-600
    if (v < 0.8) return '#F59E0B'; // amber-500
    return '#DC2626'; // red-600
  };
  const color = getColor(clampedValue);
  const strokeWidth = size * 0.1;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - clampedValue * circumference * 0.75; // 75% circle gauge
  return (
    <div
      className="flex flex-col items-center justify-center"
      style={{
        width: size
      }}>

      <div
        className="relative"
        style={{
          width: size,
          height: size
        }}>

        {/* Background Circle */}
        <svg width={size} height={size} className="transform rotate-[135deg]">
          <circle
            stroke="#E5E7EB"
            strokeWidth={strokeWidth}
            fill="transparent"
            r={radius}
            cx={size / 2}
            cy={size / 2}
            strokeDasharray={circumference}
            strokeDashoffset={circumference * 0.25}
            strokeLinecap="round" />

        </svg>

        {/* Foreground Circle */}
        <svg
          width={size}
          height={size}
          className="absolute top-0 left-0 transform rotate-[135deg]">

          <motion.circle
            stroke={color}
            strokeWidth={strokeWidth}
            fill="transparent"
            r={radius}
            cx={size / 2}
            cy={size / 2}
            strokeDasharray={circumference}
            strokeDashoffset={circumference} // Start empty
            animate={{
              strokeDashoffset: offset
            }}
            transition={{
              duration: 1.5,
              ease: 'easeOut'
            }}
            strokeLinecap="round" />

        </svg>

        {/* Value Text */}
        {showValue &&
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-4">
            <span className="text-2xl font-bold text-gray-900">
              {clampedValue.toFixed(2)}
            </span>
            <span className="text-xs text-gray-500">ε used</span>
          </div>
        }
      </div>
      {label &&
      <p className="mt-2 text-sm font-medium text-gray-600">{label}</p>
      }
    </div>);

}