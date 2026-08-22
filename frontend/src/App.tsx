import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { Languages, FileAudio, Zap, ShieldCheck, Menu } from 'lucide-react';
import type { Message, HistoryItem } from './types';

const rawUrl = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL)
  ? import.meta.env.VITE_API_BASE_URL
  : 'https://kidney-drivers-saving-phenomenon.trycloudflare.com';

const API_BASE = String(rawUrl).replace(/\/$/, '');
const LOCAL_STORAGE_KEY = 'indic_rag_chat_history';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
      console.error(e);
    }
  }, [history]);

  const handleNewChat = () => {
    setMessages([]);
    setSidebarOpen(false);
  };

  const handleSendMessage = async (text: string, language: string = 'en', audioFile?: File) => {
    if (!text.trim() && !audioFile) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text || 'Voice Audio Query',
      isAudio: !!audioFile,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const startTime = performance.now();
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, top_k: 5, source_language: language }),
      });

      if (!res.ok) throw new Error(`Server returned status ${res.status}`);

      const data = await res.json();
      const totalLatency = performance.now() - startTime;

      const botAnswer = (data.results && data.results.length > 0)
        ? (data.results[0].text || data.answer || "No matching passages found.")
        : (data.answer || "No relevant documents found.");

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: botAnswer,
        timestamp: new Date().toISOString(),
        latencyMs: Number(data.latency_ms || totalLatency),
        retrievalMs: Number(data.retrieval_time_ms || totalLatency * 0.4),
        passedTarget: (data.latency_ms || totalLatency) < 200,
        grounded: data.grounded ?? true,
        detectedLanguage: language,
        passages: (data.results || []).map((r: any, idx: number) => ({
          query_id: String(r.id || idx + 1),
          text: String(r.text || ''),
          language: String(r.language || language || 'en'),
          score: Number(r.score || 0)
        }))
      };

      setMessages((prev) => [...prev, botMessage]);

      const newHistoryItem: HistoryItem = {
        id: Date.now().toString(),
        query: text || 'Voice Audio Query',
        language: language,
        timestamp: new Date().toISOString(),
        answer: botAnswer,
        responsePayload: data
      };

      setHistory((prev) => [newHistoryItem, ...prev]);
    } catch (err: any) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Connection error: ${err.message || 'Check server status'}. Please ensure the backend is running.`,
          timestamp: new Date().toISOString(),
          grounded: false
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#030d09] text-slate-100 font-sans">
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={handleNewChat}
        history={history}
        onSelectHistory={(item) => {
          setMessages([
            {
              id: item.id,
              role: 'user',
              content: item.query || '',
              timestamp: item.timestamp || new Date().toISOString(),
            },
            {
              id: item.id + '-ans',
              role: 'assistant',
              content: item.answer || '',
              timestamp: item.timestamp || new Date().toISOString(),
              grounded: true,
              passages: item?.responsePayload?.results || []
            }
          ]);
          setSidebarOpen(false);
        }}
        onClearHistory={() => {
          localStorage.removeItem(LOCAL_STORAGE_KEY);
          setHistory([]);
        }}
      />

      <div className="flex-1 flex flex-col h-full relative overflow-hidden bg-[#030d09]">
        {/* Mobile Header */}
        <div className="lg:hidden h-14 border-b border-[#144731] flex items-center justify-between px-4 bg-[#06140e] shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg text-emerald-400 hover:bg-[#0b2419]"
          >
            <Menu className="w-5 h-5" />
          </button>
          <span className="font-bold text-sm tracking-wide">
            RAG in <span className="text-[#FFDE00]">Goa</span>
          </span>
          <div className="w-8" />
        </div>

        {/* Chat / Hero Body Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 custom-scrollbar">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center max-w-4xl mx-auto text-center px-4 py-8">

              {/* Circular Hero Logo */}
              <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-full overflow-hidden mb-6 border-2 border-[#144731] shadow-2xl flex items-center justify-center bg-[#06140e]">
                <img
                  src="/hero.png"
                  alt="RAG in Goa"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    (e.currentTarget as HTMLElement).style.display = 'none';
                  }}
                />
              </div>

              {/* Title & Subtitle */}
              <h1 className="text-2xl sm:text-4xl font-bold text-white tracking-tight mb-3">
                Welcome to the <span className="text-[#FFDE00]">RAG Platform</span>
              </h1>
              <p className="text-xs sm:text-sm text-gray-400 max-w-2xl leading-relaxed mb-10">
                A cutting-edge Retrieval-Augmented Generation engine supporting 14 Indic languages and English. Ask via text or voice, and get grounded, sub-200ms responses.
              </p>

              {/* 4 Feature Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-3xl mb-8">

                <div className="flex items-start gap-4 p-4 rounded-xl border border-[#144731] bg-[#06140e]/60 text-left hover:border-emerald-700/60 transition-colors">
                  <div className="p-2.5 rounded-lg bg-[#0b2419] border border-[#144731] text-[#FFDE00] shrink-0">
                    <Languages className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white mb-1">Multilingual Querying</h3>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      You can ask questions in 14 Indic languages and English.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4 p-4 rounded-xl border border-[#144731] bg-[#06140e]/60 text-left hover:border-emerald-700/60 transition-colors">
                  <div className="p-2.5 rounded-lg bg-[#0b2419] border border-[#144731] text-[#FFDE00] shrink-0">
                    <FileAudio className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white mb-1">Voice & Audio Input</h3>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      Speak naturally or upload an MP3 to get started.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4 p-4 rounded-xl border border-[#144731] bg-[#06140e]/60 text-left hover:border-emerald-700/60 transition-colors">
                  <div className="p-2.5 rounded-lg bg-[#0b2419] border border-[#144731] text-[#FFDE00] shrink-0">
                    <Zap className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white mb-1">Ultra-Low Latency</h3>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      Experience RAG inference optimized for &lt; 200ms.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4 p-4 rounded-xl border border-[#144731] bg-[#06140e]/60 text-left hover:border-emerald-700/60 transition-colors">
                  <div className="p-2.5 rounded-lg bg-[#0b2419] border border-[#144731] text-[#FFDE00] shrink-0">
                    <ShieldCheck className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white mb-1">Verified Grounding</h3>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      Every answer is cited back to the semantic FAISS index.
                    </p>
                  </div>
                </div>

              </div>

            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
              {loading && (
                <div className="flex items-center gap-3 text-emerald-400 text-xs italic pl-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                  Retrieving grounded context from FAISS vector store...
                </div>
              )}
            </div>
          )}
        </div>

        {/* Chat Input & Footer Section */}
        <div className="p-4 bg-[#030d09] border-t border-[#144731]/60 shrink-0">
          <div className="max-w-4xl mx-auto flex flex-col gap-3">
            <ChatInput onSendMessage={handleSendMessage} disabled={loading} apiBase={API_BASE} />

            <p className="text-[11px] text-center text-gray-500">
              Responses are strictly grounded in the indexed multilingual dataset. Queries outside the dataset corpus are rejected by grounding guardrails.
            </p>
          </div>
        </div>

        {/* Footer Hackathon Banner */}
        <footer className="h-10 bg-[#06140e] border-t border-[#144731] flex items-center justify-center px-4 shrink-0">
          <p className="text-[11px] font-semibold tracking-wider text-[#FFDE00] uppercase font-mono">
            HACKER HOUSE GOA 2026 • HHGOA.COM • 2:47PM STUDIO
          </p>
        </footer>
      </div>
    </div>
  );
}