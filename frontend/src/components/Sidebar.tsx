import React from 'react';
import { MessageSquarePlus, History, Database, Activity, Trash2, X } from 'lucide-react';
import type { HistoryItem } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  history: HistoryItem[];
  onSelectHistory: (item: HistoryItem) => void;
  onClearHistory: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose, onNewChat, history, onSelectHistory, onClearHistory }) => {
  const formatTimeAgo = (isoString: string) => {
    const diff = (new Date().getTime() - new Date(isoString).getTime()) / 1000;
    if (diff < 60) return "Just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  return (
    <>
      {/* 1. Mobile Dimmed Backdrop (Closes drawer on tap) */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 lg:hidden transition-opacity duration-300"
          onClick={onClose}
        />
      )}

      {/* 2. Responsive Sidebar Container */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-50
          w-72 sm:w-80 h-full bg-[#06140e] border-r border-[#144731]
          flex flex-col flex-none
          transform transition-transform duration-300 ease-in-out
          ${isOpen ? "translate-x-0" : "-translate-x-full"}
          lg:translate-x-0
        `}
      >
        {/* Mobile-Only Close Header Button */}
        <div className="lg:hidden flex items-start justify-between p-6 pb-6">
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-full overflow-hidden flex items-center justify-center">
              <img src="/hero.png" alt="Logo" className="w-full h-full object-cover rounded-full" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">RAG in <span className="text-[#FFDE00]">Goa</span></h1>
              <p className="text-xs text-emerald-400 font-medium">Multilingual Model</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-emerald-950/60 border border-emerald-900/40 mt-1"
          >
            <X size={18} />
          </button>
        </div>

      {/* Header */}
      <div className="hidden lg:block p-6 pb-6 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex-shrink-0 w-10 h-10 rounded-full overflow-hidden flex items-center justify-center">
            <img 
              src="/hero.png" 
              alt="RAG in Goa" 
              className="w-full h-full object-cover rounded-full" 
            />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight">RAG in <span className="text-[#FFDE00]">Goa</span></h1>
            <p className="text-xs text-emerald-400 font-medium">Multilingual Model</p>
          </div>
        </div>
      </div>

      <div className="px-4 pb-4 lg:px-6 lg:pb-6 border-b border-[#144731]/60 flex-shrink-0">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 bg-[#0b2419] border border-emerald-700/60 hover:border-[#FFDE00] text-slate-200 hover:text-[#FFDE00] px-4 py-2.5 rounded-lg transition-all group"
        >
          <MessageSquarePlus className="w-4 h-4 text-emerald-400 group-hover:text-[#FFDE00]" />
          <span className="text-sm font-medium">New Chat</span>
        </button>
      </div>

      {/* History */}
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        <div className="flex items-center gap-2 text-xs font-semibold text-emerald-500 uppercase tracking-wider mb-3 px-2">
          <History className="w-3.5 h-3.5" />
          Recent Queries
        </div>
        
        <div className="space-y-1">
          {history.length === 0 ? (
            <div className="text-xs text-gray-500 px-2 py-4 italic text-center">No recent queries.</div>
          ) : (
            history.map((item) => (
              <button 
                key={item.id}
                onClick={() => {
                  onSelectHistory(item);
                  onClose();
                }}
                className="w-full text-left p-2 hover:bg-[#0b2419] rounded-lg transition-colors group flex flex-col gap-1"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[10px] bg-emerald-950/40 text-[#FFDE00] font-semibold px-1.5 rounded font-mono border border-emerald-800/60 uppercase">
                    {item.language === "auto" ? (item.responsePayload.detected_language || "auto").substring(0, 2) : item.language}
                  </span>
                  <span className="text-sm text-gray-300 group-hover:text-white truncate">
                    {item.transcribedText || item.query}
                  </span>
                </div>
                <span className="text-[10px] text-gray-500 pl-8">
                  {formatTimeAgo(item.timestamp)}
                </span>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Clear History & System Status */}
      <div className="flex-shrink-0">
        <div className="px-4 pb-4">
          <button 
            onClick={onClearHistory}
            className="w-full flex items-center justify-center gap-2 py-2 text-xs font-medium text-gray-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear History
          </button>
        </div>
        
        <div className="p-4 border-t border-[#144731]/60 bg-[#06140e]/50">
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <Database className="w-3.5 h-3.5" />
                <span>FAISS Index</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-[10px] text-emerald-400 font-medium tracking-wide">ONLINE</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <Activity className="w-3.5 h-3.5" />
                <span>Latency Target</span>
              </div>
              <span className="border border-[#FFDE00]/40 text-[#FFDE00] bg-[#FFDE00]/10 px-2 py-0.5 rounded text-[11px] font-bold">
                &lt; 200ms
              </span>
            </div>
          </div>
        </div>
      </div>
      </aside>
    </>
  );
};
