export enum Language {
  English = 'English',
  Spanish = 'Spanish',
  Chinese = 'Chinese (Mandarin)',
  Hindi = 'Hindi',
  Arabic = 'Arabic',
  Portuguese = 'Portuguese',
  Bengali = 'Bengali',
  Russian = 'Russian',
  Japanese = 'Japanese',
  French = 'French',
}

export interface ExampleSentence {
  target: string;
  native: string;
}

export interface DictionaryEntry {
  term: string;
  definition: string;
  examples: ExampleSentence[];
  usageNote: string;
  imageUrl?: string;
  id: string; // Unique ID for notebook
  timestamp: number;
}

export interface NotebookItem extends DictionaryEntry {}

export type ViewMode = 'search' | 'notebook' | 'study';

// API Response Types
export interface DefinitionResponse {
  definition: string;
  examples: ExampleSentence[];
  usageNote: string;
}
