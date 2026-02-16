import React, { useId } from 'react';
import { ChevronDown } from 'lucide-react';
interface SelectOption {
  value: string;
  label: string;
}
interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: SelectOption[];
  error?: string;
  helperText?: string;
}
export function Select({
  label,
  options,
  error,
  helperText,
  className = '',
  id,
  ...props
}: SelectProps) {
  const selectId = id || useId();
  return (
    <div className="w-full">
      {label &&
      <label
        htmlFor={selectId}
        className="block text-sm font-medium text-gray-700 mb-1">

          {label}
        </label>
      }
      <div className="relative">
        <select
          id={selectId}
          className={`
            block w-full rounded-lg border-gray-300 shadow-sm 
            focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm
            appearance-none pr-10
            ${error ? 'border-red-300 focus:ring-red-500 focus:border-red-500' : ''}
            ${className}
          `}
          {...props}>

          {options.map((option) =>
          <option key={option.value} value={option.value}>
              {option.label}
            </option>
          )}
        </select>
        <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
          <ChevronDown className="h-4 w-4 text-gray-500" />
        </div>
      </div>
      {error && <p className="mt-1 text-sm text-red-600">{error}</p>}
      {helperText && !error &&
      <p className="mt-1 text-sm text-gray-500">{helperText}</p>
      }
    </div>);

}