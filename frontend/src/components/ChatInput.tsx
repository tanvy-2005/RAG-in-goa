import React, { useState, useRef, useEffect } from 'react';
import { Mic, ArrowUp, Paperclip, Globe, X, Square, Loader2 } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { TooltipProvider, ActionTooltip } from './ui/tooltip';
import { LANGUAGES } from '../types';
import type { LanguageOption } from '../types';

interface ChatInputProps {
  onSendText: (text: string, language: string) => void;
  onSendAudio: (file: File, language: string) => void;
  isLoading: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSendText, onSendAudio, isLoading }) => {
  const [text, setText] = useState('');
  const [language, setLanguage] = useState('auto');
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [text]);

  // Recording Timer
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setRecordingTime(0);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
        stream.getTracks().forEach(track => track.stop());
        onSendAudio(file, language);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone access denied", err);
      alert("Microphone access denied. Please allow microphone permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const validExtensions = ['.mp3', '.wav', '.ogg', '.m4a', '.mpeg'];
      const isAudioType = file.type.startsWith('audio/');
      const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

      if (isAudioType || hasValidExtension) {
        setAudioFile(file);
      } else {
        alert("The selected file format is not accepted. Please upload an MP3 or valid audio file (.mp3, .wav, .m4a, .ogg).");
      }
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSubmit = () => {
    if (isLoading || isRecording) return;
    
    if (audioFile) {
      onSendAudio(audioFile, language);
      setAudioFile(null);
      setText('');
    } else if (text.trim()) {
      onSendText(text.trim(), language);
      setText('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const canSend = (text.trim().length > 0 || audioFile !== null) && !isLoading && !isRecording;

  return (
    <TooltipProvider>
      <div className="w-[95%] sm:w-full max-w-3xl 2xl:max-w-4xl mx-auto px-2.5 sm:px-4 pt-2 pb-3">
      {/* Selected Audio File Chip */}
      {audioFile && (
        <div className="flex items-center justify-between bg-[#10b981]/10 border border-[#10b981] px-3 py-1.5 rounded-lg w-fit mx-auto mb-2">
          <span className="text-xs text-[#FFDE00] font-mono truncate max-w-[200px]">{audioFile.name}</span>
          <button onClick={() => setAudioFile(null)} className="ml-2 text-[#10b981] hover:text-[#FFDE00] transition-colors">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <div className="flex items-center gap-3 px-4 py-3 bg-[#0b2419]/90 backdrop-blur-md border border-[#144731] shadow-2xl rounded-full max-w-3xl mx-auto transition-all focus-within:border-[#FFDE00]/50 focus-within:ring-1 focus-within:ring-[#FFDE00]/20">
        
        {/* Left Action Group */}
        <div className="flex items-center gap-1.5 shrink-0">
          <ActionTooltip label="Select Language" side="top">
            <div>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger className="p-2 sm:p-2.5 rounded-full text-emerald-400 hover:text-emerald-300 hover:bg-emerald-950/60 border border-emerald-800/40 transition-colors flex items-center justify-center relative shadow-none focus:ring-0 h-auto w-auto flex-shrink-0">
                  <Globe className="w-5 h-5" />
                  {language !== 'auto' && <span className="absolute top-0 right-0 w-2 h-2 bg-emerald-400 rounded-full" />}
                </SelectTrigger>
                <SelectContent className="bg-[#0b2419] border-[#144731] text-white">
                  {LANGUAGES.map((lang: LanguageOption) => (
                    <SelectItem key={lang.value} value={lang.value} className="focus:bg-[#106941] focus:text-white">
                      {lang.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </ActionTooltip>
          
          <input 
            type="file" 
            accept="audio/*,.mp3,.wav,.ogg,.m4a,.mpeg" 
            onChange={handleFileChange} 
            ref={fileInputRef}
            className="hidden" 
          />
          <ActionTooltip label="Upload Audio File" side="top">
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="p-2 sm:p-2.5 rounded-full text-gray-400 hover:text-emerald-400 hover:bg-emerald-950/60 transition-colors flex items-center justify-center border border-transparent flex-shrink-0"
            >
              <Paperclip className="w-5 h-5" />
            </button>
          </ActionTooltip>
        </div>

          {/* Center Input Field */}
          <div className="flex-1 relative flex items-center min-w-0">
            {isRecording ? (
              <div className="flex items-center justify-center gap-1.5 sm:gap-3 w-full sm:w-fit border border-[#10b981] bg-[#10b981]/10 rounded-lg px-2 sm:px-4 py-1 mx-auto overflow-hidden">
                <div className="flex items-center gap-1 shrink-0">
                  <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse" />
                  <div className="w-1.5 h-3 bg-red-400 rounded-full animate-pulse delay-75" />
                  <div className="w-1.5 h-4 bg-red-500 rounded-full animate-pulse delay-150" />
                  <div className="w-1.5 h-2 bg-red-400 rounded-full animate-pulse delay-75" />
                </div>
                <span className="text-xs sm:text-sm font-mono text-[#FFDE00] whitespace-nowrap overflow-hidden text-ellipsis">Recording... {formatTime(recordingTime)}</span>
              </div>
            ) : (
              <div className="relative flex-1 w-full flex items-center">
                {text.length === 0 && (
                  <div className="absolute left-2 right-2 top-0 bottom-0 flex items-center pointer-events-none">
                    <span className="text-slate-500 text-[10px] sm:text-sm font-medium leading-[1.2] line-clamp-2 sm:line-clamp-1">
                      Ask anything in 14 Indic languages and English
                    </span>
                  </div>
                )}
                <input
                  type="text"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit();
                    }
                  }}
                  placeholder=""
                  className="flex-1 min-w-0 w-full bg-transparent border-0 outline-none text-left text-slate-100 text-xs sm:text-sm font-medium focus:ring-0 px-2 relative z-10"
                  disabled={isLoading}
                />
              </div>
            )}
          </div>

        {/* Right Action Group */}
        <div className="flex items-center gap-1.5 shrink-0">
          <ActionTooltip label={isRecording ? "Stop Recording" : "Record Voice"} side="top">
            {isRecording ? (
              <button 
                onClick={stopRecording}
                className="p-2 sm:p-2.5 bg-red-500/20 text-red-500 hover:bg-red-500 hover:text-white rounded-full transition-colors flex items-center justify-center flex-shrink-0"
              >
                <Square className="w-5 h-5 fill-current" />
              </button>
            ) : (
              <button 
                onClick={startRecording}
                disabled={isLoading || audioFile !== null}
                className="p-2 sm:p-2.5 text-gray-400 hover:text-emerald-400 hover:bg-emerald-950/60 rounded-full transition-colors disabled:opacity-30 flex items-center justify-center border border-transparent flex-shrink-0"
              >
                <Mic className="w-5 h-5" />
              </button>
            )}
          </ActionTooltip>

          {isLoading ? (
            <div className="p-2 sm:p-2.5 bg-[#106941] text-white rounded-full flex items-center justify-center flex-shrink-0">
              <Loader2 className="w-5 h-5 animate-spin" />
            </div>
          ) : (
            <ActionTooltip label="Send Query" side="top">
              <button 
                onClick={handleSubmit}
                disabled={!canSend}
                className={`p-2 sm:p-2.5 rounded-full transition-all flex items-center justify-center flex-shrink-0 ${
                  canSend 
                    ? "bg-[#FFDE00] hover:bg-[#ffe53b] text-slate-950 shadow-lg shadow-[#FFDE00]/20 hover:scale-105" 
                    : "bg-[#144731]/50 text-gray-500 cursor-not-allowed"
                }`}
              >
                <ArrowUp className="w-5 h-5 stroke-[2.5]" />
              </button>
            </ActionTooltip>
          )}
        </div>
      </div>
      
    </div>
    </TooltipProvider>
  );
};
