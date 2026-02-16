import React from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
interface Column<T> {
  header: string;
  accessor: keyof T | ((item: T) => React.ReactNode);
  sortable?: boolean;
  className?: string;
}
interface TableProps<T> {
  data: T[];
  columns: Column<T>[];
  onSort?: (key: keyof T, direction: 'asc' | 'desc') => void;
  sortKey?: keyof T;
  sortDirection?: 'asc' | 'desc';
  onRowClick?: (item: T) => void;
}
export function Table<
  T extends {
    id: string | number;
  }>(
{
  data,
  columns,
  onSort,
  sortKey,
  sortDirection,
  onRowClick
}: TableProps<T>) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((column, index) =>
            <th
              key={index}
              scope="col"
              className={`px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${column.sortable ? 'cursor-pointer hover:bg-gray-100' : ''} ${column.className || ''}`}
              onClick={() => {
                if (
                column.sortable &&
                onSort &&
                typeof column.accessor === 'string')
                {
                  const newDirection =
                  sortKey === column.accessor && sortDirection === 'asc' ?
                  'desc' :
                  'asc';
                  onSort(column.accessor as keyof T, newDirection);
                }
              }}>

                <div className="flex items-center space-x-1">
                  <span>{column.header}</span>
                  {column.sortable &&
                sortKey === column.accessor && (
                sortDirection === 'asc' ?
                <ChevronUp className="w-4 h-4" /> :

                <ChevronDown className="w-4 h-4" />)
                }
                </div>
              </th>
            )}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((item) =>
          <tr
            key={item.id}
            onClick={() => onRowClick && onRowClick(item)}
            className={
            onRowClick ?
            'cursor-pointer hover:bg-gray-50 transition-colors' :
            ''
            }>

              {columns.map((column, colIndex) =>
            <td
              key={colIndex}
              className={`px-6 py-4 whitespace-nowrap text-sm text-gray-900 ${column.className || ''}`}>

                  {typeof column.accessor === 'function' ?
              column.accessor(item) :
              item[column.accessor] as React.ReactNode}
                </td>
            )}
            </tr>
          )}
        </tbody>
      </table>
    </div>);

}