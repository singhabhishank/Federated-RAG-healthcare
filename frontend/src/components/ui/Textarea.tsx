import React, { useId } from 'react';
interface TextareaProps extends
  React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  helperText?: string;
}
export function Textarea({
  label,
  error,
  helperText,
  className = '',
  id,
  ...props
}: TextareaProps) {
  const textareaId = id || useId();
  return (
    <div className="w-full">
      {label &&
      <label
        htmlFor={textareaId}
        className="block text-sm font-medium text-gray-700 mb-1">

          {label}
        </label>
      }
      <textarea
        id={textareaId}
        className={`
          block w-full rounded-lg border-gray-300 shadow-sm 
          focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm
          disabled:bg-gray-50 disabled:text-gray-500
          ${error ? 'border-red-300 focus:ring-red-500 focus:border-red-500' : ''}
          ${className}
        `}
        {...props} />

      {error && <p className="mt-1 text-sm text-red-600">{error}</p>}
      {helperText && !error &&
      <p className="mt-1 text-sm text-gray-500">{helperText}</p>
      }
    </div>);

}