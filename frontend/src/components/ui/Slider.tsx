import React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';
interface SliderProps {
  value: number[];
  onValueChange: (value: number[]) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  formatValue?: (value: number) => string;
}
export function Slider({
  value,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  formatValue = (v) => v.toString()
}: SliderProps) {
  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-2">
        {label &&
        <label className="text-sm font-medium text-gray-700">{label}</label>
        }
        <span className="text-sm text-gray-500 font-mono">
          {formatValue(value[0])}
        </span>
      </div>
      <SliderPrimitive.Root
        className="relative flex items-center select-none touch-none w-full h-5"
        value={value}
        onValueChange={onValueChange}
        max={max}
        min={min}
        step={step}>

        <SliderPrimitive.Track className="bg-gray-200 relative grow rounded-full h-[3px]">
          <SliderPrimitive.Range className="absolute bg-indigo-600 rounded-full h-full" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          className="block w-5 h-5 bg-white border border-gray-300 shadow-sm rounded-full hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-transform hover:scale-110"
          aria-label="Volume" />

      </SliderPrimitive.Root>
    </div>);

}