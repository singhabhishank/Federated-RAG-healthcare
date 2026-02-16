import React from 'react';
interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  width?: string | number;
  height?: string | number;
  circle?: boolean;
}
export function Skeleton({
  width,
  height,
  circle = false,
  className = '',
  ...props
}: SkeletonProps) {
  return (
    <div
      className={`bg-gray-200 animate-pulse ${circle ? 'rounded-full' : 'rounded'} ${className}`}
      style={{
        width: width,
        height: height
      }}
      {...props} />);


}