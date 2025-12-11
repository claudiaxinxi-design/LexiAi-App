import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Language, DictionaryEntry, ViewMode } from './types';
import { getDefinition, generateImage } from './services/geminiService';
import { loadNotebook, saveNotebook } from './services/storageService';
import { ResultCard } from './components/ResultCard';
import { Notebook } from './components/Notebook';
import { StudyMode } from './components/StudyMode';

const App = () => {
  // State
  // Hardcoded for Spanish Dictionary for English Speakers
  const nativeLang = Language.English;
  const targetLang = Language.Spanish;

  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentResult, setCurrentResult] = useState<DictionaryEntry | null>(null);
  
  // Initialize from storage immediately to prevent race conditions or data loss
  const [savedItems, setSavedItems] = useState<DictionaryEntry[]>(() => loadNotebook());
  
  const [view, setView] = useState<ViewMode>('search');

  // Save items when updated
  useEffect(() => {
    saveNotebook(savedItems);
  }, [savedItems]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchTerm.trim()) return;

    setLoading(true);
    setCurrentResult(null);
    setView('search');

    try {
      // Parallel execution for speed
      const defPromise = getDefinition(searchTerm, nativeLang, targetLang);
      const imgPromise = generateImage(searchTerm, targetLang);

      const [defRes, imgRes] = await Promise.all([defPromise, imgPromise]);

      const newEntry: DictionaryEntry = {
        id: Date.now().toString(),
        term: searchTerm,
        definition: defRes.definition,
        examples: defRes.examples,
        usageNote: defRes.usageNote,
        imageUrl: imgRes || undefined,
        timestamp: Date.now(),
      };

      setCurrentResult(newEntry);
    } catch (error) {
      console.error(error);
      alert("Oops! Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const toggleSave = (entry: DictionaryEntry) => {
    const exists = savedItems.find(i => i.term === entry.term && i.definition === entry.definition);
    if (exists) {
      setSavedItems(prev => prev.filter(i => i.id !== exists.id));
    } else {
      setSavedItems(prev => [entry, ...prev]);
    }
  };

  const removeItem = (id: string) => {
    setSavedItems(prev => prev.filter(i => i.id !== id));
  };

  // Check if current result is saved
  const isCurrentSaved = currentResult 
    ? savedItems.some(i => i.term === currentResult.term && i.definition === currentResult.definition) 
    : false;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 pb-20">
      
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b-2 border-pop-dark px-4 py-3 flex justify-between items-center shadow-sm">
        <div className="flex items-center gap-2" onClick={() => setView('search')}>
            <div className="w-8 h-8 bg-pop-yellow rounded-full border-2 border-pop-dark"></div>
            <h1 className="text-2xl font-display font-black text-pop-dark cursor-pointer tracking-tight">Lexi<span className="text-pop-pink">Ai</span></h1>
        </div>
        <div className="flex gap-2">
           <button 
             onClick={() => setView('search')}
             className={`p-2 rounded-lg font-bold text-sm ${view === 'search' ? 'bg-pop-dark text-white' : 'text-gray-500'}`}
           >
             Search
           </button>
           <button 
             onClick={() => setView('notebook')}
             className={`p-2 rounded-lg font-bold text-sm ${view === 'notebook' ? 'bg-pop-dark text-white' : 'text-gray-500'}`}
           >
             Notebook
           </button>
           <button 
             onClick={() => setView('study')}
             className={`p-2 rounded-lg font-bold text-sm ${view === 'study' ? 'bg-pop-dark text-white' : 'text-gray-500'}`}
           >
             Study
           </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 pt-6 max-w-3xl">
        
        {view === 'search' && (
          <div className="animate-fade-in">
            {/* Search Setup */}
            <div className="bg-white p-6 rounded-3xl border-2 border-pop-dark shadow-[8px_8px_0px_0px_rgba(255,215,0,1)] mb-8">
              <div className="mb-2 ml-1">
                 <span className="text-xs font-bold uppercase tracking-wider text-gray-400">
                    Spanish <span className="text-gray-300 mx-1">→</span> English
                 </span>
              </div>

              <form onSubmit={handleSearch} className="relative">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Type a word in Spanish..."
                  className="w-full bg-gray-50 border-2 border-gray-200 rounded-xl px-4 py-4 pr-12 text-lg font-bold placeholder-gray-400 focus:outline-none focus:border-pop-pink focus:bg-white transition-all"
                />
                <button
                  type="submit"
                  disabled={loading || !searchTerm}
                  className="absolute right-2 top-2 bottom-2 bg-pop-pink hover:bg-pink-500 text-white rounded-lg px-4 font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                      <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                  ) : (
                    'GO'
                  )}
                </button>
              </form>
            </div>

            {/* Results Area */}
            {currentResult && !loading && (
              <ResultCard 
                entry={currentResult} 
                onSave={toggleSave} 
                isSaved={isCurrentSaved} 
                nativeLang={nativeLang}
                targetLang={targetLang}
              />
            )}
            
            {!currentResult && !loading && (
                <div className="text-center mt-20 opacity-50">
                    <div className="text-6xl mb-4 grayscale">🚀</div>
                    <p className="font-bold text-xl">Ready to learn Spanish?</p>
                </div>
            )}
          </div>
        )}

        {view === 'notebook' && (
          <Notebook 
            savedItems={savedItems} 
            nativeLang={nativeLang}
            targetLang={targetLang}
            onRemove={removeItem} 
          />
        )}

        {view === 'study' && (
          <StudyMode items={savedItems} />
        )}

      </main>
    </div>
  );
};

export default App;