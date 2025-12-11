import React, { useState } from 'react';
import { DictionaryEntry } from '../types';
import { AudioButton } from './AudioButton';
import { getQuickAiAnswer } from '../services/geminiService';

interface Props {
  entry: DictionaryEntry;
  onSave: (entry: DictionaryEntry) => void;
  isSaved: boolean;
  nativeLang: string;
  targetLang: string;
}

export const ResultCard: React.FC<Props> = ({ entry, onSave, isSaved, nativeLang, targetLang }) => {
  const [isVibeExpanded, setIsVibeExpanded] = useState(false);
  const [isQnaSectionOpen, setIsQnaSectionOpen] = useState(false);
  
  // Q&A State
  const [qnaAnswer, setQnaAnswer] = useState<string | null>(null);
  const [qnaLoading, setQnaLoading] = useState(false);
  const [activeQnaType, setActiveQnaType] = useState<'natural' | 'mistake' | 'funfact' | null>(null);

  const handleQnaClick = async (type: 'natural' | 'mistake' | 'funfact') => {
    setActiveQnaType(type);
    setQnaLoading(true);
    setQnaAnswer(null); // Clear previous answer while loading
    
    try {
      const answer = await getQuickAiAnswer(entry.term, type, nativeLang, targetLang);
      setQnaAnswer(answer);
    } catch (e) {
      setQnaAnswer("Oops! My brain froze. Try again?");
    } finally {
      setQnaLoading(false);
    }
  };

  return (
    <div className="w-full bg-white border-2 border-pop-dark rounded-3xl p-6 shadow-[8px_8px_0px_0px_rgba(255,105,180,1)] mb-8 animate-fade-in-up">
      
      {/* Header: Word + Audio + Save */}
      <div className="flex justify-between items-start mb-6">
        <div className="flex flex-col gap-2">
            <div className="flex items-center gap-3">
                 <h2 className="text-4xl font-display font-black text-pop-dark">{entry.term}</h2>
                 <AudioButton text={entry.term} size="lg" />
            </div>
            {/* Tagline/Definition */}
            <p className="text-xl text-[#1c1c1c] font-bold">{entry.definition}</p>
        </div>
        
        <button
          onClick={() => onSave(entry)}
          className={`p-3 rounded-xl border-2 border-pop-dark transition-all ${
            isSaved 
            ? 'bg-pop-yellow shadow-inner' 
            : 'bg-white hover:bg-gray-50 shadow-[4px_4px_0px_0px_rgba(45,55,72,1)] active:translate-y-[2px] active:shadow-[2px_2px_0px_0px_rgba(45,55,72,1)]'
          }`}
          aria-label="Save to Notebook"
        >
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className={`h-6 w-6 ${isSaved ? 'fill-pop-dark' : 'fill-none stroke-pop-dark'}`} 
            viewBox="0 0 24 24" 
            strokeWidth="2"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
          </svg>
        </button>
      </div>

      {/* Image */}
      {entry.imageUrl ? (
        <div className="w-full h-64 bg-gray-100 rounded-xl mb-6 border-2 border-pop-dark overflow-hidden flex items-center justify-center relative">
          <img 
            src={entry.imageUrl} 
            alt={entry.term} 
            className="w-full h-full object-cover"
          />
        </div>
      ) : (
          <div className="w-full h-64 bg-gray-100 rounded-xl mb-6 border-2 border-pop-dark border-dashed flex items-center justify-center text-gray-400 font-bold">
            No Image Generated
          </div>
      )}

      {/* Usage Note - Collapsible */}
      <div className="bg-pop-yellow/20 p-4 rounded-xl border-2 border-pop-yellow mb-6">
        <div className="flex items-center gap-2 mb-2">
            <span className="text-2xl">💡</span>
            <h3 className="font-black text-pop-dark uppercase tracking-wide text-sm">The Vibe Check</h3>
        </div>
        
        <div className={`relative transition-all duration-300 ease-in-out ${isVibeExpanded ? '' : 'max-h-20 overflow-hidden'}`}>
            <p className="text-pop-dark font-medium leading-relaxed">{entry.usageNote}</p>
            {!isVibeExpanded && (
                <div className="absolute bottom-0 left-0 w-full h-12 bg-gradient-to-t from-[#FFF7CC] to-transparent pointer-events-none" />
            )}
        </div>

        <div className="flex justify-center mt-2">
            <button 
                onClick={() => setIsVibeExpanded(!isVibeExpanded)}
                className="p-1 rounded-full hover:bg-black/5 transition-colors text-pop-dark/60 hover:text-pop-dark"
                aria-label={isVibeExpanded ? "Show less" : "Show more"}
            >
                <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className={`h-6 w-6 transition-transform duration-300 ${isVibeExpanded ? 'rotate-180' : ''}`}
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor" 
                    strokeWidth="2.5"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
            </button>
        </div>
      </div>

      {/* Examples */}
      <div className="space-y-4 mb-6">
        <h3 className="font-black text-pop-dark uppercase tracking-wide text-sm ml-1">Examples</h3>
        {entry.examples.map((ex, idx) => (
          <div key={idx} className="bg-gray-50 p-4 rounded-xl border border-gray-200">
             <div className="flex justify-between items-start gap-3">
                <p className="font-bold text-pop-dark text-lg">{ex.target}</p>
                <div className="shrink-0 mt-1">
                    <AudioButton text={ex.target} size="sm" />
                </div>
             </div>
             <p className="text-gray-500 italic mt-1">{ex.native}</p>
          </div>
        ))}
      </div>

      {/* AI Q&A Section - Enhanced Visuals */}
      <div className="mt-8 border-t border-dashed border-gray-200 pt-6">
        <button 
          onClick={() => setIsQnaSectionOpen(!isQnaSectionOpen)}
          className={`w-full flex items-center justify-between p-4 rounded-2xl border-2 transition-all duration-300 group ${
            isQnaSectionOpen 
            ? 'bg-pop-purple/5 border-pop-purple shadow-none' 
            : 'bg-white border-gray-200 hover:border-pop-purple/50 shadow-sm hover:shadow-md'
          }`}
        >
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-xl transition-colors ${
               isQnaSectionOpen ? 'bg-pop-purple text-white' : 'bg-gray-100 text-gray-500 group-hover:bg-pop-purple group-hover:text-white'
            }`}>
                <span className="text-2xl">🧠</span>
            </div>
            <div className="text-left">
                <h3 className={`font-black uppercase tracking-wide text-sm transition-colors ${
                    isQnaSectionOpen ? 'text-pop-purple' : 'text-gray-700 group-hover:text-pop-purple'
                }`}>
                    AI Deep Dive
                </h3>
                {!isQnaSectionOpen && (
                    <p className="text-xs text-gray-400 font-bold mt-0.5">
                        Tap to unlock pro tips & fun facts
                    </p>
                )}
            </div>
          </div>
           <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ${isQnaSectionOpen ? 'bg-pop-purple/10 text-pop-purple rotate-180' : 'bg-gray-100 text-gray-400 group-hover:bg-pop-purple/10 group-hover:text-pop-purple'}`}>
              <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  className="h-5 w-5"
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor" 
                  strokeWidth="2.5"
              >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
           </div>
        </button>

        <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isQnaSectionOpen ? 'max-h-[500px] opacity-100 mt-4' : 'max-h-0 opacity-0'}`}>
          <div className="bg-pop-purple/5 p-4 rounded-xl border border-pop-purple/20">
            <p className="text-sm text-gray-500 mb-3 font-medium">Ask the AI for details:</p>
            
            <div className="flex flex-wrap gap-2 mb-4">
               <button 
                  onClick={() => handleQnaClick('natural')}
                  className={`px-3 py-2 rounded-lg text-sm font-bold border-2 transition-all shadow-sm active:translate-y-[1px] ${activeQnaType === 'natural' ? 'bg-pop-purple text-white border-pop-purple' : 'bg-white text-pop-dark border-gray-200 hover:border-pop-purple hover:text-pop-purple'}`}
               >
                  🗣️ Natural Use
               </button>
               <button 
                  onClick={() => handleQnaClick('mistake')}
                  className={`px-3 py-2 rounded-lg text-sm font-bold border-2 transition-all shadow-sm active:translate-y-[1px] ${activeQnaType === 'mistake' ? 'bg-pop-pink text-white border-pop-pink' : 'bg-white text-pop-dark border-gray-200 hover:border-pop-pink hover:text-pop-pink'}`}
               >
                  🚫 Mistakes
               </button>
               <button 
                  onClick={() => handleQnaClick('funfact')}
                  className={`px-3 py-2 rounded-lg text-sm font-bold border-2 transition-all shadow-sm active:translate-y-[1px] ${activeQnaType === 'funfact' ? 'bg-pop-cyan text-pop-dark border-pop-cyan' : 'bg-white text-pop-dark border-gray-200 hover:border-pop-cyan hover:text-pop-cyan'}`}
               >
                  🤓 Fun Fact
               </button>
            </div>

            {(qnaLoading || qnaAnswer) && (
               <div className="bg-white rounded-xl p-4 border border-pop-purple/10 shadow-sm relative min-h-[80px]">
                  {qnaLoading ? (
                     <div className="flex items-center justify-center h-full py-2">
                        <svg className="animate-spin h-6 w-6 text-pop-purple" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                     </div>
                  ) : (
                     <div className="animate-fade-in">
                        <p className="text-pop-dark font-medium leading-relaxed text-sm">{qnaAnswer}</p>
                     </div>
                  )}
               </div>
            )}
          </div>
        </div>
      </div>

    </div>
  );
};