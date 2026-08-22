import React, { useState } from 'react';
import { CheckCircle2, AlertTriangle, ShieldCheck, ShieldAlert, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import type { RAGResponse } from '../types';

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
  const clean = code.toLowerCase().trim();
  return LANGUAGE_FULL_NAMES[clean] || code.toUpperCase();
};

interface MetricCardProps {
  response: RAGResponse;
}

export const MetricCard: React.FC<MetricCardProps> = ({ response }) => {
  const [showPassages, setShowPassages] = useState(false);
  
  const ragLatency = response.latency_ms || 0;
  const retrievalMs = response.retrieval_ms || 0;
  const audioTotalMs = response.audio_pipeline_total_ms;
  
  const isSub200 = response.passed_target_200ms === true || (response.latency_ms !== undefined && response.latency_ms < 200);
  const isGrounded = response.grounded !== false;

  return (
    <div className="bg-[#0b2419] border border-[#144731] rounded-xl overflow-hidden shadow-lg mt-6">
      <div className="p-6">
        <h2 className="text-2xl font-bold text-white mb-6">Results</h2>

        {response.transcribed_text && (
          <div className="mb-6 p-4 bg-[#06140e] rounded-lg border border-[#144731]">
            <div className="text-xs text-emerald-500 font-semibold mb-1 uppercase tracking-wider">Transcribed Audio</div>
            <p className="text-gray-300 italic">"{response.transcribed_text}"</p>
          </div>
        )}

        <div className="mb-8">
          <div className="text-xs text-emerald-500 font-semibold mb-2 uppercase tracking-wider">Main Answer</div>
          <div className="text-lg text-white leading-relaxed p-4 bg-[#072718] rounded-lg border-l-4 border-[#106941]">
            {response.answer}
          </div>
        </div>

        <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-${audioTotalMs ? '5' : '4'} gap-4 mb-6`}>
          <div className="bg-[#06140e] p-4 rounded-lg border border-[#144731] flex flex-col">
            <span className="text-xs text-emerald-500 font-semibold mb-1 uppercase tracking-wider">RAG Latency</span>
            <div className="flex items-end gap-2">
              <span className="text-2xl font-bold text-white">{ragLatency.toFixed(0)}</span>
              <span className="text-gray-400 mb-1">ms</span>
            </div>
          </div>
          
          <div className="bg-[#06140e] p-4 rounded-lg border border-[#144731] flex flex-col">
            <span className="text-xs text-emerald-500 font-semibold mb-1 uppercase tracking-wider">Vector Retrieval</span>
            <div className="flex items-end gap-2">
              <span className="text-2xl font-bold text-white">{retrievalMs.toFixed(0)}</span>
              <span className="text-gray-400 mb-1">ms</span>
            </div>
          </div>

          {audioTotalMs && (
            <div className="bg-[#06140e] p-4 rounded-lg border border-[#144731] flex flex-col">
              <span className="text-xs text-emerald-500 font-semibold mb-1 uppercase tracking-wider">Speech-to-Text Pipeline</span>
              <div className="flex items-end gap-2">
                <span className="text-2xl font-bold text-white">{(audioTotalMs / 1000).toFixed(1)}</span>
                <span className="text-gray-400 mb-1">s</span>
              </div>
            </div>
          )}
          
          <div className={`p-4 rounded-lg border flex flex-col justify-center ${isSub200 ? 'bg-emerald-950/60 border-emerald-700/50' : 'bg-amber-950/40 border-amber-900/50'}`}>
            <div className="flex items-center gap-2 mb-1">
              {isSub200 ? <CheckCircle2 className="w-5 h-5 text-emerald-400" /> : <AlertTriangle className="w-5 h-5 text-amber-500" />}
              <span className={`text-xs font-semibold uppercase tracking-wider ${isSub200 ? 'text-emerald-400' : 'text-amber-500'}`}>
                Performance Goal
              </span>
            </div>
            <span className="text-sm font-medium text-white">{isSub200 ? '✓ Target Met (<200ms)' : 'Above Target Latency'}</span>
          </div>

          <div className="bg-[#06140e] p-4 rounded-lg border border-[#144731] flex flex-col justify-center">
             <div className="flex items-center gap-2 mb-1">
              {isGrounded ? <ShieldCheck className="w-5 h-5 text-emerald-400" /> : <ShieldAlert className="w-5 h-5 text-amber-500" />}
              <span className={`text-xs font-semibold uppercase tracking-wider ${isGrounded ? 'text-emerald-400' : 'text-amber-500'}`}>
                Grounding
              </span>
            </div>
            <span className="text-sm font-medium text-white">{isGrounded ? 'Verified Grounded' : 'Ungrounded Response'}</span>
          </div>
        </div>

        {response.detected_language && (
          <div className="inline-block px-3 py-1 bg-[#106941] text-emerald-100 rounded-full text-xs font-semibold mb-6">
            Language: {getLanguageFullName(response.detected_language)}
          </div>
        )}

        <div className="border-t border-[#144731] pt-4 mt-2">
          <button 
            onClick={() => setShowPassages(!showPassages)}
            className="flex items-center justify-between w-full text-emerald-400 hover:text-emerald-300 font-medium transition-colors"
          >
            <div className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              <span>Retrieved Source Passages ({response.passages?.length || 0})</span>
            </div>
            {showPassages ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>
          
          {showPassages && (
            <div className="mt-4 space-y-3">
              {response.passages?.map((p: any, idx: number) => (
                <div key={idx} className="bg-[#06140e] p-4 rounded-lg border border-[#144731]">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs bg-[#0b2419] border border-[#144731] text-emerald-300 px-2 py-1 rounded">
                      ID: {p.id}
                    </span>
                    <span className="text-xs font-mono text-emerald-500">
                      Score: {p.similarity?.toFixed(4)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300">{p.text}</p>
                </div>
              ))}
              {(!response.passages || response.passages.length === 0) && (
                <div className="text-center text-gray-500 py-4 text-sm">No passages retrieved.</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
