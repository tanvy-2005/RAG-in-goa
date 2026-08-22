import React, { useState } from 'react';
import { Send, Loader2, Search } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { LANGUAGES } from '../types';
import type { LanguageOption } from '../types';

interface TextSearchProps {
  onSearch: (query: string, language: string) => Promise<void>;
  isLoading: boolean;
}

const SUGGESTIONS = [
  "What are the main agricultural products of Goa?",
  "How is the weather in monsoon?",
  "Goa public transport details"
];

export const TextSearch: React.FC<TextSearchProps> = ({ onSearch, isLoading }) => {
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState("auto");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query.trim(), language);
    }
  };

  return (
    <div className="bg-[#0b2419] border border-[#144731] rounded-xl p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-4">
        <Search className="text-emerald-400 w-5 h-5" />
        <h2 className="text-xl font-semibold text-white">Text Query</h2>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <Select value={language} onValueChange={setLanguage}>
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
          
          <div className="flex-1 relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question..."
              className="w-full bg-[#06140e] border border-[#144731] text-white rounded-lg px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-[#106941]"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!query.trim() || isLoading}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-[#106941] hover:bg-[#15803d] text-white rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mt-2">
          {SUGGESTIONS.map((suggestion, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setQuery(suggestion)}
              className="text-xs bg-[#072718] border border-[#144731] hover:bg-[#106941] hover:border-transparent text-emerald-300 px-3 py-1.5 rounded-full transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </form>
    </div>
  );
};
