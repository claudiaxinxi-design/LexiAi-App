import React, { useState } from 'react';
import { DictionaryEntry } from '../types';
import { AudioButton } from './AudioButton';

interface Props {
  items: DictionaryEntry[];
}

export const StudyMode: React.FC<Props> = ({ items }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);

  if (items.length === 0) {
     return (
        <div className="flex flex-col items-center justify-center h-64 text-center p-6">
            <div className="text-6xl mb-4">🎓</div>
            <h3 className="text-xl font-bold text-pop-dark mb-2">No Cards to Study</h3>
            <p className="text-gray-500">Save some words to your notebook first!</p>
        </div>
    );
  }

  const currentItem = items[currentIndex];

  const handleNext = () => {
    setIsFlipped(false);
    setTimeout(() => {
        setCurrentIndex((prev) => (prev + 1) % items.length);
    }, 200);
  };

  const handlePrev = () => {
    setIsFlipped(false);
    setTimeout(() => {
        setCurrentIndex((prev) => (prev - 1 + items.length) % items.length);
    }, 200);
  };

  return (
    <div className="flex flex-col items-center w-full max-w-md mx-auto h-[600px]">
      
      <div className="flex justify-between w-full items-center mb-6 px-4">
        <h2 className="text-2xl font-display font-black text-pop-dark">Flashcards</h2>
        <span className="font-bold text-gray-400">{currentIndex + 1} / {items.length}</span>
      </div>

      <div className="relative w-full h-[400px] cursor-pointer group perspective-1000" onClick={() => setIsFlipped(!isFlipped)}>
        <div className={`card-flip w-full h-full duration-500 preserve-3d relative ${isFlipped ? 'flipped' : ''}`}>
          
          {/* Front */}
          <div className="card-inner absolute inset-0 w-full h-full">
             <div className="card-front absolute inset-0 w-full h-full bg-white border-4 border-pop-blue rounded-3xl shadow-[8px_8px_0px_0px_rgba(65,105,225,1)] flex flex-col items-center justify-center p-6 backface-hidden z-10">
                {currentItem.imageUrl ? (
                    <img src={currentItem.imageUrl} alt="Hint" className="w-32 h-32 object-cover rounded-full border-2 border-gray-200 mb-6" />
                ) : (
                    <div className="text-6xl mb-6">❓</div>
                )}
                <h3 className="text-4xl font-black text-pop-dark mb-4">{currentItem.term}</h3>
                <p className="text-sm text-gray-400 font-bold uppercase tracking-widest">Tap to flip</p>
             </div>

             {/* Back */}
             <div className="card-back absolute inset-0 w-full h-full bg-pop-dark text-white rounded-3xl shadow-[8px_8px_0px_0px_rgba(0,0,0,0.2)] flex flex-col items-center justify-center p-6 backface-hidden transform rotate-y-180" style={{ transform: 'rotateY(180deg)' }}>
                <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-3xl font-bold text-pop-yellow">{currentItem.term}</h3>
                    <AudioButton text={currentItem.term} size="sm" variant="light" />
                </div>
                
                <p className="text-xl font-medium mb-6 text-center">{currentItem.definition}</p>
                
                <div className="w-full bg-white/10 rounded-xl p-4 text-left">
                     <p className="text-sm text-gray-300 italic mb-1">Example:</p>
                     <p className="text-white text-md">{currentItem.examples[0].target}</p>
                     <p className="text-gray-400 text-sm">{currentItem.examples[0].native}</p>
                </div>
             </div>
          </div>
        </div>
      </div>

      <div className="flex gap-6 mt-10">
        <button onClick={handlePrev} className="bg-white p-4 rounded-full border-2 border-pop-dark shadow-[4px_4px_0px_0px_rgba(45,55,72,1)] active:translate-y-[2px] active:shadow-none transition-all">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
        </button>
        <button onClick={handleNext} className="bg-pop-yellow p-4 rounded-full border-2 border-pop-dark shadow-[4px_4px_0px_0px_rgba(45,55,72,1)] active:translate-y-[2px] active:shadow-none transition-all">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
        </button>
      </div>

    </div>
  );
};