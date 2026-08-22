import React, { useState, useRef } from 'react';
import { UploadCloud, Loader2 } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { LANGUAGES } from '../types';
import type { LanguageOption } from '../types';

interface AudioUploadProps {
  onUpload: (file: File, language: string) => Promise<void>;
  isLoading: boolean;
}

export const AudioUpload: React.FC<AudioUploadProps> = ({ onUpload, isLoading }) => {
  const [language, setLanguage] = useState("auto");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSetFile = (selectedFile: File) => {
    const validExtensions = ['.mp3', '.wav', '.ogg', '.m4a', '.mpeg'];
    const isAudioType = selectedFile.type.startsWith('audio/');
    const hasValidExtension = validExtensions.some(ext => selectedFile.name.toLowerCase().endsWith(ext));

    if (isAudioType || hasValidExtension) {
      setFile(selectedFile);
    } else {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setFile(null);
      alert("The selected file format is not accepted. Please upload an MP3 or valid audio file (.mp3, .wav, .m4a, .ogg).");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = () => {
    if (file && !isLoading) {
      onUpload(file, language);
    }
  };

  return (
    <div className="bg-[#0b2419] border border-[#144731] rounded-xl p-6 shadow-lg flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <UploadCloud className="text-emerald-400 w-5 h-5" />
        <h2 className="text-xl font-semibold text-white">Upload Audio</h2>
      </div>

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
        
        <input 
          type="file" 
          accept="audio/*,.mp3,.wav,.ogg,.m4a,.mpeg" 
          onChange={handleFileChange} 
          ref={fileInputRef}
          className="hidden" 
        />
        
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`flex-1 border-2 border-dashed rounded-lg px-4 py-3 text-center transition-colors flex items-center justify-center cursor-pointer ${
            isDragging 
              ? "border-[#106941] bg-[#106941]/20 text-emerald-200" 
              : "border-[#144731] hover:border-[#106941] bg-[#072718] text-emerald-300"
          }`}
        >
          {file ? file.name : "Select or drop audio file (.mp3, .wav)"}
        </div>
      </div>

      {file && (
        <div className="flex justify-end mt-2">
          <button
            onClick={handleUpload}
            disabled={isLoading}
            className="flex items-center gap-2 bg-[#106941] hover:bg-[#15803d] text-white px-6 py-2.5 rounded-lg font-medium disabled:opacity-50 transition-colors"
          >
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Process Audio"}
          </button>
        </div>
      )}
    </div>
  );
};
