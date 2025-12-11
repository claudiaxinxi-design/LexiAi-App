import React from 'react';
import { Language } from '../types';

interface Props {
  label: string;
  selected: Language;
  onChange: (lang: Language) => void;
}

export const LanguageSelector: React.FC<Props> = ({ label, selected, onChange }) => {
  return (
    <div className="flex flex-col w-full">
      <label className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-1 ml-1">{label}</label>
      <div className="relative">
        <select
          value={selected}
          onChange={(e) => onChange(e.target.value as Language)}
          className="appearance-none w-full bg-white border-2 border-pop-dark rounded-xl px-4 py-3 text-pop-dark font-bold shadow-[4px_4px_0px_0px_rgba(45,55,72,1)] focus:outline-none focus:translate-y-[2px] focus:shadow-[2px_2px_0px_0px_rgba(45,55,72,1)] transition-all cursor-pointer"
        >
          {Object.values(Language).map((lang) => (
            <option key={lang} value={lang}>
              {lang}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-pop-dark">
          <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
            <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
          </svg>
        </div>
      </div>
    </div>
  );
};
