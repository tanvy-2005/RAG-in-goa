import React, { useState } from 'react';
import { Mic, FileAudio, ShieldCheck, ShieldAlert, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import type { Message } from '../types';

const LANGUAGE_FULL_NAMES: Record<string, string> = {
  en: "English",
  hi: "Hindi (हिन्दी)",
  bn: "Bengali (বাংলা)",
  ta: "Tamil (தமிழ்)",
  te: "Telugu (తెలుగు)",
  mr: "Marathi (मराठी)",
  gu: "Gujarati (ગુજરાતી)",
  kn: "Kannada (ಕನ್ನಡ)",
  ml: "Malayalam (മലയാളം)",
  pa: "Punjabi (ਪੰਜਾਬੀ)",
  or: "Odia (ଓଡ଼ିଆ)",
  as: "Assamese (অসমীয়া)",
  ur: "Urdu (اردو)",
  ne: "Nepali (नेपाली)",
  sa: "Sanskrit (संस्कृतम्)"
};

const getLanguageFullName = (code?: string) => {
  if (!code) return "Unknown";
  const clean = String(code).toLowerCase().trim();
  return LANGUAGE_FULL_NAMES[clean] || String(code).toUpperCase();
};

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = (message?.role === 'user') || ((message as any)?.sender === 'user');
  const [showPassages, setShowPassages] = useState(false);

  // Safe fallback for text content to prevent .split on undefined
  const contentText = String(message?.content || (message as any)?.text || '');

  if (isUser) {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-[92%] sm:max-w-[85%] md:max-w-[80%] flex flex-col items-end gap-2">
          {message?.isAudio && (
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#0b2419] border border-[#144731] text-xs text-emerald-400">
              <Mic className="w-3 h-3" />
              <span>Voice Audio</span>
            </div>
          )}
          <div className="bg-[#0b2419] border border-[#144731]/60 text-white px-5 py-3.5 rounded-2xl rounded-tr-sm shadow-sm">
            <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{contentText}</p>
          </div>
        </div>
      </div>
    );
  }

  // Safe split mapping for multiline content
  const lines = contentText.split('\n');

  return (
    <div className="flex justify-start mb-8 w-full">
      <div className="flex gap-4 w-full max-w-4xl">
        {/* Assistant Message Avatar */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full overflow-hidden flex items-center justify-center mt-1">
          <img
            src="/hero.png"
            alt="RAG in Goa"
            className="w-full h-full object-cover rounded-full"
            onError={(e) => {
              // Fallback placeholder if image fails
              (e.currentTarget as HTMLElement).style.display = 'none';
            }}
          />
        </div>

        {/* Message Content Area */}
        <div className="flex flex-col gap-3 flex-1 min-w-0">

          {/* STT / Transcribed Pill */}
          {message?.transcribedText && (
            <div className="self-start flex flex-col gap-1.5 p-3 bg-[#06140e] border border-[#144731] rounded-xl">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-500 uppercase tracking-wider">
                <FileAudio className="w-3.5 h-3.5" />
                Transcribed Query
              </div>
              <p className="text-sm text-gray-300 italic">"{message.transcribedText}"</p>
            </div>
          )}

          {/* Main Answer */}
          <div className="text-[15px] text-white leading-relaxed prose prose-invert max-w-none">
            {lines.map((line, i) => (
              <span key={i}>
                {line}
                <br />
              </span>
            ))}
          </div>

          {/* Performance Pill Bar */}
          <div className="flex flex-wrap gap-1.5 sm:gap-2 mt-2">
            <div className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-full border border-emerald-800/60 bg-emerald-950/40 text-emerald-300">
              <span>RAG Latency: <span className="text-[#FFDE00] font-bold">{message?.latencyMs ? Math.round(message.latencyMs) : '--'} ms</span></span>
              {message?.passedTarget && (
                <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse ml-1" />
              )}
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-full border border-emerald-800/60 bg-emerald-950/40 text-emerald-300">
              <span>Vector Retrieval: <span className="text-[#FFDE00] font-bold">{message?.retrievalMs ? Math.round(message.retrievalMs) : '--'} ms</span></span>
            </div>

            <div className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-full border ${message?.grounded ? 'border-emerald-800/60 bg-emerald-950/40 text-emerald-300' : 'border-amber-900/50 bg-amber-950/40 text-amber-500'}`}>
              {message?.grounded ? <ShieldCheck className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3" />}
              <span>Grounding Status: {message?.grounded ? 'Verified Grounded' : 'Ungrounded'}</span>
            </div>

            {message?.detectedLanguage && (
              <div className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-full border border-emerald-800/60 bg-emerald-950/40 text-emerald-300">
                <span>Language: {getLanguageFullName(message.detectedLanguage)}</span>
              </div>
            )}

            {message?.audioPipelineMs && (
              <div className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-full border border-emerald-800/60 bg-emerald-950/40 text-emerald-300">
                <span>Speech-to-Text Pipeline: {(message.audioPipelineMs / 1000).toFixed(1)} s</span>
              </div>
            )}
          </div>

          {/* Citations Accordion */}
          {Array.isArray(message?.passages) && message.passages.length > 0 && (
            <div className="mt-2 border border-[#144731]/60 rounded-xl overflow-hidden bg-[#06140e]/50">
              <button
                type="button"
                onClick={() => setShowPassages(!showPassages)}
                className="w-full flex items-center justify-between p-3 text-sm text-emerald-400 hover:bg-[#0b2419] transition-colors"
              >
                <div className="flex items-center gap-2 font-medium">
                  <FileText className="w-4 h-4" />
                  View {message.passages.length} Grounded Passages
                </div>
                {showPassages ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>

              {showPassages && (
                <div className="p-3 border-t border-[#144731]/60 space-y-2 bg-[#06140e]">
                  {message.passages.map((p, idx) => (
                    <div key={idx} className="p-3 rounded-lg border border-[#144731]/40 bg-[#0b2419]/30">
                      <div className="flex justify-between items-center mb-1.5">
                        <div className="flex gap-2">
                          <span className="text-[10px] bg-[#FFDE00]/10 text-[#FFDE00] px-1.5 py-0.5 rounded border border-[#FFDE00]/40">
                            ID: {p?.query_id || idx + 1}
                          </span>
                          <span className="text-[10px] bg-emerald-950/60 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-900/50 uppercase font-mono">
                            {getLanguageFullName(p?.language)}
                          </span>
                        </div>
                        <span className="text-[10px] font-mono text-emerald-500">
                          Score: <span className="text-[#FFDE00] font-mono font-semibold">{typeof p?.score === 'number' ? p.score.toFixed(4) : (p?.score || '--')}</span>
                        </span>
                      </div>
                      <p className="text-xs sm:text-sm text-gray-300 leading-relaxed">{p?.text || ''}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

        </div>
      </div>
    </div>
  );
};