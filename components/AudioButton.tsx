import React, { useState, useEffect } from 'react';
import { generateSpeech } from '../services/geminiService';
import { decode, decodeAudioData, getAudioContext } from '../services/audioUtils';

interface AudioButtonProps {
  text: string;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'light'; // 'default' for light backgrounds, 'light' for dark backgrounds
}

export const AudioButton: React.FC<AudioButtonProps> = ({ text, label, size = 'md', variant = 'default' }) => {
  const [loading, setLoading] = useState(false);
  const [cachedBuffer, setCachedBuffer] = useState<AudioBuffer | null>(null);

  // Reset cache if text changes
  useEffect(() => {
    setCachedBuffer(null);
  }, [text]);

  const playAudio = (buffer: AudioBuffer, ctx: AudioContext) => {
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start();
  };

  const handlePlay = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (loading) return;

    const ctx = getAudioContext();
    // Resume context if suspended (browser policy)
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    if (cachedBuffer) {
      playAudio(cachedBuffer, ctx);
      return;
    }

    setLoading(true);
    try {
      const base64 = await generateSpeech(text);
      if (base64) {
        const audioBuffer = await decodeAudioData(
          decode(base64),
          ctx
        );
        setCachedBuffer(audioBuffer);
        playAudio(audioBuffer, ctx);
      }
    } catch (err) {
      console.error("Audio playback error", err);
    } finally {
      setLoading(false);
    }
  };

  const iconSize = size === 'sm' ? 'w-4 h-4' : size === 'lg' ? 'w-8 h-8' : 'w-6 h-6';
  
  const baseClasses = `inline-flex items-center gap-2 justify-center rounded-full transition-all active:scale-95 ${size === 'sm' ? 'p-1.5' : 'p-2'}`;
  
  const variantClasses = variant === 'light'
    ? 'bg-white/20 hover:bg-white/30 text-pop-yellow'
    : 'bg-pop-cyan/20 hover:bg-pop-cyan/40 text-pop-dark';

  const spinnerColor = variant === 'light' ? 'text-pop-yellow' : 'text-pop-purple';

  return (
    <button
      onClick={handlePlay}
      disabled={loading}
      className={`${baseClasses} ${variantClasses}`}
      title="Play pronunciation"
    >
      {loading ? (
        <svg className={`animate-spin ${iconSize} ${spinnerColor}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      ) : (
        <svg className={`${iconSize} fill-current`} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
          <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
        </svg>
      )}
      {label && <span className="text-sm font-semibold">{label}</span>}
    </button>
  );
};