import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { BrainCircuit, Languages, Menu } from 'lucide-react';
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

  const handleSendMessage = async (text: string, language: string = 'en') => {
    if (!text.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: text,
      language: language || 'en',
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, top_k: 5, source_language: language }),
      });

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

      const data = await res.json();
      const botAnswer = (data.results && data.results.length > 0)
        ? data.results[0].text
        : "No relevant documents found.";

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: botAnswer,
        language: language || 'en',
        timestamp: new Date().toISOString(),
        ragResponse: {
          query: text,
          language: language || 'en',
          answer: botAnswer,
          grounded: true,
          passages: (data.results || []).map((r: any) => ({
            id: String(r.id || ''),
            text: String(r.text || ''),
            language: String(r.language || 'en'),
            score: Number(r.score || 0)
          })),
          latency_ms: Number(data.latency_ms || 0),
          retrieval_count: (data.results || []).length
        }
      };

      setMessages((prev) => [...prev, botMessage]);

      const newHistoryItem: HistoryItem = {
        id: Date.now().toString(),
        query: text,
        language: language || 'en',
        timestamp: new Date().toISOString(),
        answer: botAnswer,
        responsePayload: botMessage.ragResponse!
      };

      setHistory((prev) => [newHistoryItem, ...prev]);
    } catch (err: any) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          sender: 'bot',
          text: `Error connecting to backend: ${err.message || 'Check server status'}.`,
          language: 'en',
          timestamp: new Date().toISOString(),
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        history={history}
        onSelectHistory={(item) => {
          setMessages([
            {
              id: item.id,
              sender: 'user',
              text: item.query || '',
              language: item.language || 'en',
              timestamp: item.timestamp || new Date().toISOString(),
            },
            {
              id: item.id + '-ans',
              sender: 'bot',
              text: item.answer || '',
              language: item.language || 'en',
              timestamp: item.timestamp || new Date().toISOString(),
              ragResponse: item.responsePayload
            }
          ]);
          setSidebarOpen(false);
        }}
        onClearHistory={() => {
          localStorage.removeItem(LOCAL_STORAGE_KEY);
          setHistory([]);
        }}
      />

      <div className="flex-1 flex flex-col h-full relative overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50 backdrop-blur shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-6 h-6 text-emerald-400" />
              <h1 className="font-bold text-lg text-slate-100 tracking-tight">
                Indic Multilingual RAG
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-950/50 border border-emerald-800/60 px-3 py-1.5 rounded-full font-mono">
            <Languages className="w-3.5 h-3.5" />
            <span>22 Indic Languages Ready</span>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto text-slate-400 space-y-4">
              <div className="p-4 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl">
                <BrainCircuit className="w-12 h-12 text-emerald-400 mx-auto" />
              </div>
              <h2 className="text-xl font-bold text-slate-200">Multilingual Knowledge Base</h2>
              <p className="text-sm text-slate-400">
                Ask questions across Indian languages. Semantic search is powered by Indic FAISS embeddings.
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))
          )}
          {loading && (
            <div className="flex items-center gap-3 text-slate-400 text-sm italic pl-2">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              Retrieving relevant context from vector database...
            </div>
          )}
        </div>

        {/* Input */}
        <div className="p-4 bg-slate-900/60 border-t border-slate-800 shrink-0">
          <ChatInput onSendMessage={handleSendMessage} disabled={loading} apiBase={API_BASE} />
        </div>
      </div>
    </div>
  );
}