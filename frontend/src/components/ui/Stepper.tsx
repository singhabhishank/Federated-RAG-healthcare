import React from 'react';
import { Check, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
export interface Step {
  id: string;
  title: string;
  description?: string;
  status: 'pending' | 'current' | 'completed' | 'error';
  icon?: React.ReactNode;
}
interface StepperProps {
  steps: Step[];
  orientation?: 'horizontal' | 'vertical';
}
export function Stepper({ steps, orientation = 'horizontal' }: StepperProps) {
  return (
    <div
      className={`flex ${orientation === 'vertical' ? 'flex-col space-y-8' : 'flex-row justify-between items-start w-full'}`}>

      {steps.map((step, index) =>
      <div
        key={step.id}
        className={`flex ${orientation === 'vertical' ? 'flex-row items-start' : 'flex-col items-center flex-1 relative'}`}>

          {/* Connector Line for Horizontal */}
          {orientation === 'horizontal' && index < steps.length - 1 &&
        <div className="absolute top-4 left-1/2 w-full h-0.5 bg-gray-200 -z-10">
              <motion.div
            className="h-full bg-indigo-600"
            initial={{
              width: '0%'
            }}
            animate={{
              width: step.status === 'completed' ? '100%' : '0%'
            }}
            transition={{
              duration: 0.5
            }} />

            </div>
        }

          {/* Icon/Indicator */}
          <div
          className={`
            flex items-center justify-center w-8 h-8 rounded-full border-2 transition-colors duration-300 z-10 bg-white
            ${step.status === 'completed' ? 'border-indigo-600 bg-indigo-600 text-white' : ''}
            ${step.status === 'current' ? 'border-indigo-600 text-indigo-600' : ''}
            ${step.status === 'pending' ? 'border-gray-300 text-gray-400' : ''}
            ${step.status === 'error' ? 'border-red-600 text-red-600' : ''}
          `}>

            {step.status === 'completed' ?
          <Check className="w-5 h-5" /> :
          step.status === 'current' ?
          <Loader2 className="w-5 h-5 animate-spin" /> :

          step.icon ||
          <span className="text-xs font-medium">{index + 1}</span>

          }
          </div>

          {/* Content */}
          <div
          className={`
            ${orientation === 'vertical' ? 'ml-4' : 'mt-2 text-center'}
          `}>

            <h3
            className={`text-sm font-medium ${step.status === 'current' ? 'text-indigo-600' : 'text-gray-900'}`}>

              {step.title}
            </h3>
            {step.description &&
          <p className="text-xs text-gray-500 mt-0.5">{step.description}</p>
          }
          </div>

          {/* Connector Line for Vertical */}
          {orientation === 'vertical' && index < steps.length - 1 &&
        <div className="absolute left-4 top-8 bottom-[-32px] w-0.5 bg-gray-200 -ml-[1px]">
              <motion.div
            className="w-full bg-indigo-600"
            initial={{
              height: '0%'
            }}
            animate={{
              height: step.status === 'completed' ? '100%' : '0%'
            }}
            transition={{
              duration: 0.5
            }} />

            </div>
        }
        </div>
      )}
    </div>);

}