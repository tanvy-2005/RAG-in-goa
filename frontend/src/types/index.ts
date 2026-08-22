export type LanguageOption = {
  value: string;
  label: string;
};

export const LANGUAGES: LanguageOption[] = [
  { value: "auto", label: "Auto-Detect" },
  { value: "hi", label: "Hindi" },
  { value: "en", label: "English" },
  { value: "bn", label: "Bengali" },
  { value: "ta", label: "Tamil" },
  { value: "te", label: "Telugu" },
  { value: "mr", label: "Marathi" },
  { value: "gu", label: "Gujarati" },
  { value: "as", label: "Assamese" },
  { value: "kn", label: "Kannada" },
  { value: "ml", label: "Malayalam" },
  { value: "pa", label: "Punjabi" },
  { value: "or", label: "Odia" },
  { value: "ur", label: "Urdu" },
  { value: "ne", label: "Nepali" },
  { value: "sa", label: "Sanskrit" }
];

export interface Passage {
  score: number;
  language: string;
  query_id: number | string;
  text: string;
}

export interface RAGResponse {
  query: string;
  language: string;
  answer: string;
  grounded: boolean;
  passages: Passage[];
  latency_ms: number;
  retrieval_ms: number;
  passed_target_200ms: boolean;
  transcribed_text?: string;
  detected_language?: string;
  audio_pipeline_total_ms?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isAudio?: boolean; // For user message
  
  // Assistant specific metadata (mapped from QueryResponse)
  passages?: Passage[];
  latencyMs?: number;
  retrievalMs?: number;
  audioPipelineMs?: number;
  detectedLanguage?: string;
  transcribedText?: string;
  grounded?: boolean;
  passedTarget?: boolean;
}

export interface HistoryItem {
  id: string;
  query: string;
  language: string;
  timestamp: string;
  transcribedText?: string;
  answer: string;
  responsePayload: RAGResponse;
}
