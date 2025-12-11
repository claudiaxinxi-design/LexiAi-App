import React, { useState } from 'react';
import { DictionaryEntry, Language } from '../types';
import { generateStory } from '../services/geminiService';

interface Props {
  savedItems: DictionaryEntry[];
  nativeLang: Language;
  targetLang: Language;
  onRemove: (id: string) => void;
}

export const Notebook: React.FC<Props> = ({ savedItems, nativeLang, targetLang, onRemove }) => {
  const [story, setStory] = useState<string | null>(null);
  const [loadingStory, setLoadingStory] = useState(false);

  const handleMakeStory = async () => {
    if (savedItems.length === 0) return;
    setLoadingStory(true);
    try {
        const words = savedItems.map(i => i.term);
        const result = await generateStory(words, nativeLang, targetLang);
        setStory(result);
    } catch (e) {
        console.error(e);
    } finally {
        setLoadingStory(false);
    }
  };

  if (savedItems.length === 0) {
    return (
        <div className="flex flex-col items-center justify-center h-64 text-center p-6">
            <div className="text-6xl mb-4">📓</div>
            <h3 className="text-xl font-bold text-pop-dark mb-2">Notebook is Empty</h3>
            <p className="text-gray-500">Search for words and save them here to study later!</p>
        </div>
    );
  }

  return (
    <div className="w-full max-w-2xl mx-auto pb-24">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-display font-black text-pop-dark">My Words ({savedItems.length})</h2>
        <button 
            onClick={handleMakeStory}
            disabled={loadingStory}
            className="bg-pop-purple text-white font-bold py-2 px-4 rounded-xl border-2 border-pop-dark shadow-[4px_4px_0px_0px_rgba(45,55,72,1)] active:translate-y-[2px] active:shadow-[2px_2px_0px_0px_rgba(45,55,72,1)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
            {loadingStory ? 'Thinking...' : '✨ Surprise Me'}
        </button>
      </div>

      {story && (
        <div className="bg-pop-pink/10 border-2 border-pop-pink rounded-xl p-6 mb-8 relative">
            <button 
                onClick={() => setStory(null)} 
                className="absolute top-2 right-2 text-pop-pink hover:bg-pop-pink/20 rounded-full p-1"
            >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
            <h3 className="text-pop-pink font-bold uppercase tracking-wider text-sm mb-3">AI Mini Dialogue</h3>
            <p className="text-lg text-pop-dark whitespace-pre-line leading-relaxed font-medium">{story}</p>
        </div>
      )}

      <div className="space-y-4">
        {savedItems.map((item) => (
            <div key={item.id} className="bg-white p-4 rounded-xl border-2 border-gray-200 flex justify-between items-center group hover:border-pop-blue transition-colors">
                <div>
                    <div className="flex items-center gap-2">
                        <h3 className="font-bold text-xl text-pop-dark">{item.term}</h3>
                         <span className="text-sm text-gray-400 px-2 py-0.5 rounded-md bg-gray-100">{item.definition}</span>
                    </div>
                </div>
                <button 
                    onClick={() => onRemove(item.id)}
                    className="text-gray-300 hover:text-red-500 p-2"
                >
                     <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                </button>
            </div>
        ))}
      </div>
    </div>
  );
};