import React, { useState, useRef } from 'react';
import { Mic, Square, Loader2, AlertCircle } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { LANGUAGES } from '../types';
import type { LanguageOption } from '../types';

interface AudioRecorderProps {
  onRecordComplete: (blob: Blob, language: string) => Promise<void>;
  isLoading: boolean;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordComplete, isLoading }) => {
  const [language, setLanguage] = useState("auto");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        onRecordComplete(audioBlob, language);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error("Microphone access denied or error:", err);
      setError("Microphone access denied. Please grant permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-[#0b2419] border border-[#144731] rounded-xl p-6 shadow-lg flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <Mic className="text-emerald-400 w-5 h-5" />
        <h2 className="text-xl font-semibold text-white">Live Microphone</h2>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-red-400 bg-red-950/40 p-3 rounded-lg border border-red-900/50">
          <AlertCircle className="w-5 h-5" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      <div className="flex flex-col sm:flex-row gap-4 items-center">
        <Select value={language} onValueChange={setLanguage} disabled={isRecording || isLoading}>
          <SelectTrigger className="w-full sm:w-48">
            <SelectValue placeholder="Language" />
          </SelectTrigger>
          <SelectContent>
            {LANGUAGES.map((lang: LanguageOption) => (
              <SelectItem key={lang.value} value={lang.value}>
                {lang.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        
        <div className="flex-1 flex justify-center items-center gap-4">
          {!isRecording ? (
            <button
              onClick={startRecording}
              disabled={isLoading}
              className="flex items-center gap-2 bg-[#106941] hover:bg-[#15803d] text-white px-6 py-3 rounded-full font-medium disabled:opacity-50 transition-colors shadow-[0_0_15px_rgba(16,105,65,0.4)]"
            >
              <Mic className="w-5 h-5" /> Record Audio
            </button>
          ) : (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-emerald-400 bg-emerald-950/50 px-4 py-2 rounded-full border border-[#106941]">
                <div className="w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse" />
                <span className="font-mono">{formatTime(recordingTime)}</span>
              </div>
              <button
                onClick={stopRecording}
                className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-full font-medium transition-colors"
              >
                <Square className="w-5 h-5 fill-current" /> Stop
              </button>
            </div>
          )}
          
          {isLoading && !isRecording && (
            <div className="flex items-center gap-2 text-emerald-400">
              <Loader2 className="w-5 h-5 animate-spin" /> Processing...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
