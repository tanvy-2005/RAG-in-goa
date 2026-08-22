import React, { useState, useRef, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { BrainCircuit, Languages, FileAudio, Zap, Menu, MessageSquarePlus, Plus } from 'lucide-react';
import type { Message, RAGResponse, HistoryItem } from './types';

const API_BASE = 'http://127.0.0.1:8000/api';
const LOCAL_STORAGE_KEY = 'indic_rag_chat_history';

const DEFAULT_HISTORY: HistoryItem[] = [
  {
    id: "hist-1",
    query: "கேள்வி: ஒரு நிறுவனம் என்பது என்ன?",
    language: "ta",
    timestamp: new Date(Date.now() - 500000).toISOString(),
    answer: "ஒரு நிறுவனம் என்பது சட்டபூர்வமான தனி நபர் அந்தஸ்து கொண்ட அமைப்பாகும்.",
    responsePayload: { query: "", language: "ta", answer: "", grounded: true, passages: [], latency_ms: 90, retrieval_ms: 15, passed_target_200ms: true, detected_language: "ta" }
  },
  {
    id: "hist-2",
    query: "ଏକ କର୍ପୋରେସନ୍ କ'ଣ?",
    language: "or",
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    answer: "କର୍ପୋରେସନ୍ ହେଉଛି ଏକ ବ୍ୟବସାୟିକ ସଂଗଠନ ଯାହା ଆଇନ ଅନୁଯାୟୀ ଏକ ସ୍ୱତନ୍ତ୍ର ବ୍ୟକ୍ତି ଭାବରେ ସ୍ୱୀକୃତିପ୍ରାପ୍ତ |",
    responsePayload: { query: "", language: "or", answer: "", grounded: true, passages: [], latency_ms: 120, retrieval_ms: 22, passed_target_200ms: true, detected_language: "or" }
  },
  {
    id: "hist-3",
    query: "ഒരു കോർപ്പറേഷൻ എന്താണ്?",
    language: "ml",
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    answer: "നിയമപരമായി അംഗീകരിക്കപ്പെട്ട ഒരു കൂട്ടം വ്യക്തികളോ കമ്പനിയോ ഉൾപ്പെടുന്ന ഒരു സ്ഥാപനമാണ് കോർപ്പറേഷൻ.",
    responsePayload: { query: "", language: "ml", answer: "", grounded: true, passages: [], latency_ms: 110, retrieval_ms: 18, passed_target_200ms: true, detected_language: "ml" }
  },
  {
    id: "hist-4",
    query: "किमुच्यते बर्टेली?",
    language: "sa",
    timestamp: new Date(Date.now() - 86400000).toISOString(),
    answer: "बर्टेली इत्यस्य विषये अधिकं विवरणं न प्राप्तम्।",
    responsePayload: { query: "", language: "sa", answer: "", grounded: false, passages: [], latency_ms: 105, retrieval_ms: 20, passed_target_200ms: true, detected_language: "sa" }
  },
  {
    id: "hist-5",
    query: "के हो निगम?",
    language: "ne",
    timestamp: new Date(Date.now() - 172800000).toISOString(),
    transcribedText: "के हो निगम?",
    answer: "निगम भनेको कानूनी रूपमा मान्यता प्राप्त एक व्यावसायिक संस्था हो।",
    responsePayload: { query: "", language: "ne", answer: "", grounded: true, passages: [], latency_ms: 150, retrieval_ms: 30, passed_target_200ms: true, detected_language: "ne", audio_pipeline_total_ms: 1100 }
  }
];

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (stored) {
      try {
        setHistory(JSON.parse(stored));
      } catch (e) {
        setHistory(DEFAULT_HISTORY);
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(DEFAULT_HISTORY));
      }
    } else {
      setHistory(DEFAULT_HISTORY);
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(DEFAULT_HISTORY));
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const saveToHistory = (item: HistoryItem) => {
    const newHistory = [item, ...history];
    setHistory(newHistory);
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(newHistory));
  };

  const handleNewChat = () => {
    setMessages([]);
    setIsMobileMenuOpen(false);
  };

  const handleSelectHistory = (item: HistoryItem) => {
    const userMsg: Message = {
      id: item.id + '-user',
      role: 'user',
      content: item.query,
      timestamp: new Date(item.timestamp),
      isAudio: !!item.transcribedText,
    };
    const aiMsg: Message = {
      id: item.id + '-ai',
      role: 'assistant',
      content: item.answer,
      timestamp: new Date(item.timestamp),
      passages: item.responsePayload.passages,
      latencyMs: item.responsePayload.latency_ms,
      retrievalMs: item.responsePayload.retrieval_ms,
      audioPipelineMs: item.responsePayload.audio_pipeline_total_ms,
      transcribedText: item.transcribedText || item.responsePayload.transcribed_text,
      grounded: item.responsePayload.grounded,
      passedTarget: item.responsePayload.passed_target_200ms,
      detectedLanguage: item.responsePayload.detected_language || item.language,
    };
    setMessages([userMsg, aiMsg]);
    setIsMobileMenuOpen(false);
  };

  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  };

  const handleSendText = async (text: string, language: string) => {
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, language }),
      });
      
      const data: RAGResponse = await res.json();
      
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        passages: data.passages,
        latencyMs: data.latency_ms,
        retrievalMs: data.retrieval_ms,
        grounded: data.grounded,
        passedTarget: data.passed_target_200ms,
        detectedLanguage: data.detected_language || data.language || language,
      };
      
      setMessages(prev => [...prev, aiMsg]);

      // Save to history
      saveToHistory({
        id: Date.now().toString(),
        query: text,
        language: data.detected_language || language,
        timestamp: new Date().toISOString(),
        answer: data.answer,
        responsePayload: data
      });

    } catch (error) {
      console.error(error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I encountered an error trying to connect to the server. Please try again.",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendAudio = async (file: File, language: string) => {
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: 'Uploaded an audio file.',
      timestamp: new Date(),
      isAudio: true,
    };
    
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', language);

    try {
      const res = await fetch(`${API_BASE}/voice-ask`, {
        method: 'POST',
        body: formData,
      });
      
      const data: RAGResponse = await res.json();
      
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        passages: data.passages,
        latencyMs: data.latency_ms,
        retrievalMs: data.retrieval_ms,
        audioPipelineMs: data.audio_pipeline_total_ms,
        transcribedText: data.transcribed_text,
        grounded: data.grounded,
        passedTarget: data.passed_target_200ms,
        detectedLanguage: data.detected_language || data.language || language,
      };
      
      setMessages(prev => [...prev, aiMsg]);

      // Save to history
      saveToHistory({
        id: Date.now().toString(),
        query: data.transcribed_text || "Audio File",
        language: data.detected_language || language,
        timestamp: new Date().toISOString(),
        transcribedText: data.transcribed_text,
        answer: data.answer,
        responsePayload: data
      });

    } catch (error) {
      console.error(error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I encountered an error trying to process the audio file. Please try again.",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#05130c] font-sans overflow-hidden">
      <Sidebar 
        isOpen={isMobileMenuOpen}
        onClose={() => setIsMobileMenuOpen(false)}
        onNewChat={handleNewChat} 
        history={history}
        onSelectHistory={handleSelectHistory}
        onClearHistory={handleClearHistory}
      />
      
      <main className="flex-1 flex flex-col relative">
        {/* Mobile Top Navbar Header */}
        <header className="lg:hidden flex items-center justify-between px-4 py-3 bg-[#06140e]/95 backdrop-blur-md border-b border-[#144731] z-30 flex-none sticky top-0">
          {/* Left: Hamburger Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(true)}
            className="p-2 rounded-lg text-emerald-400 hover:text-emerald-300 hover:bg-emerald-950/60 border border-emerald-800/40 transition-colors"
            aria-label="Open sidebar"
          >
            <Menu size={20} />
          </button>

          {/* Center: Branding */}
          <div className="flex items-center gap-2">
            <img src="/hero.png" alt="Logo" className="w-7 h-7 rounded-full object-cover" />
            <span className="text-sm font-bold text-white tracking-wide font-sans">
              RAG in <span className="text-[#FFDE00]">Goa</span>
            </span>
          </div>

          {/* Right: Invisible Spacer to Keep Center Logo Perfectly Balanced */}
          <div className="w-9 h-9 opacity-0 pointer-events-none" aria-hidden="true" />
        </header>

        {/* Message Feed */}
        <div className="flex-1 overflow-y-auto custom-scrollbar scroll-smooth">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center min-h-full px-4 pt-8 text-center max-w-3xl mx-auto pb-64">
              <div className="flex flex-col items-center justify-center text-center max-w-2xl mx-auto mt-auto pb-2 mb-10">
                {/* Compact Constrained Logo Badge */}
                <div className="flex-shrink-0 w-16 h-16 md:w-20 md:h-20 rounded-full overflow-hidden flex items-center justify-center mb-4 transition-transform hover:scale-105 duration-200">
                  <img src="/hero.png" alt="RAG in Goa Logo" className="w-full h-full object-cover rounded-full" />
                </div>
              
                {/* Typography */}
                <h2 className="text-2xl md:text-3xl font-bold text-white tracking-tight font-sans mb-3">
                  Welcome to the <span className="text-[#FFDE00] drop-shadow-[0_0_12px_rgba(255,222,0,0.25)]">RAG Platform</span>
                </h2>
                <p className="text-xs md:text-sm text-slate-400 max-w-lg leading-relaxed font-sans">
                  A cutting-edge Retrieval-Augmented Generation engine supporting 14 Indic languages and English. Ask via text or voice, and get grounded, sub-200ms responses.
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 max-w-3xl mx-auto w-full px-2 sm:px-0 mb-auto">
                {[
                  { icon: <Languages className="w-5 h-5" />, title: "Multilingual Querying", desc: "You can ask questions in 14 Indic languages and English.", iconBg: "bg-[#FFDE00]/10 border-[#FFDE00]/30 text-[#FFDE00]", hover: "hover:border-[#FFDE00]/60 hover:shadow-[0_0_20px_rgba(255,222,0,0.08)]" },
                  { icon: <FileAudio className="w-5 h-5" />, title: "Voice & Audio Input", desc: "Speak naturally or upload an MP3 to get started.", iconBg: "bg-[#FFDE00]/10 border-[#FFDE00]/30 text-[#FFDE00]", hover: "hover:border-[#FFDE00]/60 hover:shadow-[0_0_20px_rgba(255,222,0,0.08)]" },
                  { icon: <Zap className="w-5 h-5" />, title: "Ultra-Low Latency", desc: "Experience RAG inference optimized for < 200ms.", iconBg: "bg-[#FFDE00]/10 border-[#FFDE00]/30 text-[#FFDE00]", hover: "hover:border-[#FFDE00]/60 hover:shadow-[0_0_20px_rgba(255,222,0,0.08)]" },
                  { icon: <BrainCircuit className="w-5 h-5" />, title: "Verified Grounding", desc: "Every answer is cited back to the semantic FAISS index.", iconBg: "bg-[#FFDE00]/10 border-[#FFDE00]/30 text-[#FFDE00]", hover: "hover:border-[#FFDE00]/60 hover:shadow-[0_0_20px_rgba(255,222,0,0.08)]" }
                ].map((card, i) => (
                  <button key={i} className={`flex items-start gap-4 p-4 rounded-xl bg-[#0b2419]/70 hover:bg-[#0e2f21]/80 border border-[#144731] transition-all duration-200 shadow-sm group text-left ${card.hover}`}>
                    <div className={`p-2.5 rounded-lg border transition-colors flex-shrink-0 ${card.iconBg}`}>
                      {card.icon}
                    </div>
                    <div className="flex flex-col space-y-1">
                      <h3 className="text-sm font-semibold text-slate-100 tracking-wide">{card.title}</h3>
                      <p className="text-xs text-slate-400 font-normal leading-relaxed">{card.desc}</p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="w-full max-w-4xl 2xl:max-w-5xl mx-auto px-3 sm:px-6 lg:px-8 pt-8 pb-40 sm:pb-48 lg:pb-52">
              {messages.map(msg => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Bottom Sticky/Fixed Dock Area */}
        <div className="absolute bottom-0 left-0 w-full pointer-events-none flex flex-col items-center bg-gradient-to-t from-[#05130c] via-[#05130c]/95 to-transparent pt-12">
          
          {/* 1. Floating Input Capsule */}
          <div className="w-full pointer-events-auto">
            <ChatInput 
              onSendText={handleSendText} 
              onSendAudio={handleSendAudio} 
              isLoading={isLoading} 
            />
          </div>

          {/* 2. Guardrails Disclaimer - Perfectly Centered Between Input & Footer */}
          <div className="w-full py-3 flex items-center justify-center pointer-events-auto">
            <p className="text-slate-400/80 text-[10px] sm:text-xs text-center leading-relaxed max-w-4xl px-6 select-none">
              Responses are strictly grounded in the indexed multilingual dataset. Queries outside the dataset corpus are rejected by grounding guardrails.
            </p>
          </div>

          {/* 3. Hacker House Goa Footer */}
          <footer className="w-full py-3 bg-[#082c1b] border-t border-[#144731] flex flex-col items-center justify-center space-y-1 text-center font-sans z-10 pointer-events-auto">
            <div className="text-[#FFDE00] font-bold tracking-widest text-xs uppercase drop-shadow-sm">
              HACKER HOUSE GOA 2026
            </div>
            <div className="text-[#34d399] text-[10px] sm:text-[11px] font-medium tracking-wider uppercase opacity-90">
              HHGOA.COM • OCT 28–31, 2026 • 2:47PM STUDIO
            </div>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
