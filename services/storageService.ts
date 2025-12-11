import { DictionaryEntry } from '../types';

const STORAGE_KEY = 'lexiai_notebook';

export const loadNotebook = (): DictionaryEntry[] => {
  if (typeof window === 'undefined') return [];
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return [];
    
    // Parse and simple validation
    const parsed = JSON.parse(saved);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    return [];
  } catch (e) {
    console.error("Failed to load notebook from storage:", e);
    return [];
  }
};

export const saveNotebook = (items: DictionaryEntry[]): void => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch (e) {
    console.error("Failed to save notebook to storage:", e);
  }
};