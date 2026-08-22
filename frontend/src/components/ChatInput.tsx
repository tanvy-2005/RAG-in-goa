import React, { useState, useRef } from 'react';
import { Send, Globe, Mic, Paperclip, Loader2 } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (text: string, language: string, audioFile?: File) => void;
  disabled?: boolean;
  apiBase?: string;
}

const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'Hindi (हिन्दी)' },
  { code: 'bn', name: 'Bengali (বাংলা)' },
  { code: 'ta', name: 'Tamil (தமிழ்)' },
  { code: 'te', name: 'Telugu (తెలుగు)' },
  { code: 'mr', name: 'Marathi (मराठी)' },
  { code: 'gu', name: 'Gujarati (ગુજરાતી)' },
  { code: 'kn', name: 'Kannada (ಕನ್ನಡ)' },
  { code: 'ml', name: 'Malayalam (മലയാളം)' },
  { code: 'pa', name: 'Punjabi (ਪੰਜਾਬੀ)' },
  { code: 'or', name: 'Odia (ଓଡ଼ିଆ)' },
  { code: 'as', name: 'Assamese (অসমীয়া)' },
  { code: 'ur', name: 'Urdu (اردو)' },
  { code: 'ne', name: 'Nepali (नेपाली)' },
  { code: 'sa', name: 'Sanskrit (संस्कृतम्)' },
];

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled = false }) => {
  const [input, setInput] = useState('');
  const [selectedLang, setSelectedLang] = useState('en');
  const [showLangMenu, setShowLangMenu] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || disabled) return;
    onSendMessage(input.trim(), selectedLang);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onSendMessage(file.name, selectedLang, file);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative w-full max-w-4xl mx-auto">
      {/* Language Popup */}
      {showLangMenu && (
        <div className="absolute bottom-16 left-2 z-50 bg-[#06140e] border border-[#144731] rounded-xl p-2 shadow-2xl grid grid-cols-2 sm:grid-cols-3 gap-1.5 w-72 sm:w-96 max-h-60 overflow-y-auto">
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              type="button"
              onClick={() => {
                setSelectedLang(lang.code);
                setShowLangMenu(false);
              }}
              className={`px-2.5 py-1.5 rounded text-left text-xs font-medium transition-colors ${selectedLang === lang.code
                  ? 'bg-emerald-800 text-[#FFDE00] font-bold'
                  : 'text-gray-300 hover:bg-[#0b2419] hover:text-white'
                }`}
            >
              {lang.name}
            </button>
          ))}
        </div>
      )}

      {/* Main Bar */}
      <div className="flex items-center gap-2 bg-[#06140e] border-2 border-emerald-900/60 focus-within:border-emerald-500 rounded-full px-3 py-2 shadow-xl transition-all">
        {/* Language selector button */}
        <button
          type="button"
          onClick={() => setShowLangMenu(!showLangMenu)}
          className="p-2 rounded-full text-emerald-400 hover:bg-[#0b2419] hover:text-[#FFDE00] transition-colors flex items-center gap-1 text-xs"
          title="Select Language"
        >
          <Globe className="w-5 h-5" />
          <span className="uppercase font-mono font-bold text-[11px] hidden sm:inline">
            {selectedLang}
          </span>
        </button>

        {/* File upload attachment */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="p-2 rounded-full text-gray-400 hover:text-white hover:bg-[#0b2419] transition-colors"
          title="Upload Audio Query"
        >
          <Paperclip className="w-4 h-4" />
        </button>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          accept="audio/*,.mp3,.wav"
          className="hidden"
        />

        {/* Input Text Box */}
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything in 14 Indic languages and English..."
          disabled={disabled}
          className="flex-1 bg-transparent text-white placeholder-gray-500 text-sm focus:outline-none px-2"
        />

        {/* Mic Indicator */}
        <button
          type="button"
          onClick={() => onSendMessage("Audio voice query recorded", selectedLang)}
          className="p-2 rounded-full text-gray-400 hover:text-emerald-400 hover:bg-[#0b2419] transition-colors"
          title="Record Audio Query"
        >
          <Mic className="w-4 h-4" />
        </button>

        {/* Send Button */}
        <button
          type="submit"
          disabled={disabled || !input.trim()}
          className="p-2.5 rounded-full bg-[#FFDE00] text-black hover:bg-yellow-400 active:scale-95 disabled:opacity-40 disabled:hover:bg-[#FFDE00] disabled:cursor-not-allowed transition-all shadow-md flex items-center justify-center"
        >
          {disabled ? (
            <Loader2 className="w-4 h-4 animate-spin text-black" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </div>
    </form>
  );
};