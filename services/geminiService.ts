import { GoogleGenAI, Type, Modality } from "@google/genai";
import { DefinitionResponse } from "../types";

const apiKey = process.env.API_KEY || ''; // Injected by environment
const ai = new GoogleGenAI({ apiKey });

// Helper to generate a consistent seed from a string
const getSeed = (str: string): number => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
};

// 1. Get Text Definition
export const getDefinition = async (
  term: string,
  nativeLang: string,
  targetLang: string
): Promise<DefinitionResponse> => {
  const prompt = `
    Define the term "${term}" (which is in ${targetLang}) for a speaker of ${nativeLang}.
    Provide:
    1. A natural language definition in ${nativeLang}.
    2. Two example sentences. CRITICAL: 
       - The 'target' field MUST be the sentence in ${targetLang}.
       - The 'native' field MUST be the translation in ${nativeLang}.
    3. A "usageNote" in ${nativeLang} that explains cultural nuance, tone, or related words. 
       CRITICAL: The usage note must be fun, lively, and casual. Like a friend talking. No textbook style. Be concise.
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
    config: {
      temperature: 0, // Enforce consistent text results
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          definition: { type: Type.STRING },
          examples: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                target: {
                  type: Type.STRING,
                  description: `The example sentence in ${targetLang} (e.g. Spanish).`
                },
                native: {
                  type: Type.STRING,
                  description: `The translation of the example in ${nativeLang} (e.g. English).`
                }
              }
            }
          },
          usageNote: { type: Type.STRING }
        }
      }
    }
  });

  if (!response.text) throw new Error("No definition generated");
  return JSON.parse(response.text) as DefinitionResponse;
};

// 2. Generate Image
export const generateImage = async (term: string, targetLang: string): Promise<string | null> => {
  try {
    const seed = getSeed(term.toLowerCase().trim());
    
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image', // Using flash-image for generation as per latest capabilities in prompt
      contents: {
        parts: [
          { text: `A simple, bright, pop-art style illustration representing the concept of "${term}" (in ${targetLang}). Minimalist, colorful, vector art style. White background. High contrast, thick lines.` }
        ]
      },
      config: {
        seed: seed
      }
    });

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    return null;
  } catch (error) {
    console.error("Image generation failed", error);
    return null; // Fallback or handle gracefully
  }
};

// 3. Generate Speech (TTS)
export const generateSpeech = async (text: string): Promise<string | null> => {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Kore' }, // Natural sounding voice
          },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    return base64Audio || null;
  } catch (error) {
    console.error("TTS failed", error);
    return null;
  }
};

// 4. Generate Story (Updated to Dialogue)
export const generateStory = async (
  words: string[],
  nativeLang: string,
  targetLang: string
): Promise<string> => {
  const prompt = `
    Create a short, practical real-life dialogue in ${targetLang} using as many of these words as possible naturally: ${words.join(', ')}.
    
    Requirements:
    1. Format as a simple conversation script (e.g., Person A: ... / Person B: ...).
    2. Sentences must be simple, short, and beginner-friendly.
    3. IMMEDIATELY after each ${targetLang} sentence, include the ${nativeLang} translation in parentheses on the same line.
    4. Keep it concise (max 6-8 lines of dialogue).
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
  });

  return response.text || "Could not generate dialogue.";
};

// 5. Get Quick AI Answer for Q&A
export const getQuickAiAnswer = async (
  term: string,
  type: 'natural' | 'mistake' | 'funfact',
  nativeLang: string,
  targetLang: string
): Promise<string> => {
  let specificPrompt = "";
  
  switch (type) {
    case 'natural':
      specificPrompt = `Explain the most natural, authentic ways to use the word "${term}" in casual conversation. Give a couple of quick examples.`;
      break;
    case 'mistake':
      specificPrompt = `What are the most common mistakes learners make when using or pronouncing "${term}"? How can they avoid them?`;
      break;
    case 'funfact':
      specificPrompt = `Tell me a fun fact, etymology, or a "tricky" mnemonic way to memorize "${term}".`;
      break;
  }

  const fullPrompt = `
    You are a fun, energetic language tutor. 
    The user is asking about the word "${term}" (which is in ${targetLang}).
    Answer this request for a ${nativeLang} speaker: "${specificPrompt}"
    
    CRITICAL INSTRUCTIONS:
    1. Keep the answer VERY SHORT (1-4 sentences maximum).
    2. Be simple and beginner-friendly.
    3. Use casual language and emojis.
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: fullPrompt,
    config: {
      temperature: 0, // Enforce consistent answers for the same prompt
    }
  });

  return response.text || "Could not generate an answer right now.";
};