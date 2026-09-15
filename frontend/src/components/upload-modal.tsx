"use client";

import { useState, useRef } from "react";
import { Upload, X, File, CheckCircle, Loader2 } from "lucide-react";

interface UploadModalProps {
  onClose: () => void;
  onUploadComplete: (documentId: string) => void;
}

export function UploadModal({ onClose, onUploadComplete }: UploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progressStage, setProgressStage] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const STAGES = [
    "Slicing pages...",
    "Extracting tables...",
    "Generating context...",
    "Indexing vectors...",
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setProgressStage(STAGES[0]);

    // Simulate progress stages updating since backend might not stream progress
    // In a real app, you'd use Server-Sent Events or WebSockets for true progress.
    let stageIndex = 0;
    const progressInterval = setInterval(() => {
      stageIndex++;
      if (stageIndex < STAGES.length) {
        setProgressStage(STAGES[stageIndex]);
      } else {
        clearInterval(progressInterval);
      }
    }, 15000); // Assume each pass takes ~15s for the simulation UI

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8000/api/v1/documents/upload", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (res.ok) {
        const data = await res.json();
        onUploadComplete(data.document_id);
      } else {
        throw new Error("Upload failed");
      }
    } catch (e) {
      console.error(e);
      alert("Failed to upload document.");
      setIsUploading(false);
      clearInterval(progressInterval);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden animate-in fade-in zoom-in duration-200">
        <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Upload Custom Report</h2>
          <button 
            onClick={onClose}
            disabled={isUploading}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full text-gray-500 transition-colors disabled:opacity-50"
          >
            <X size={20} />
          </button>
        </div>

        <div className="p-6">
          {!isUploading ? (
            <div className="space-y-4">
              <div 
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all"
              >
                <div className="bg-blue-100 dark:bg-blue-900/50 p-3 rounded-full text-blue-600 dark:text-blue-400">
                  <Upload size={24} />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">Click to browse or drag and drop</p>
                  <p className="text-xs text-gray-500 mt-1">PDF financial reports only</p>
                </div>
                <input 
                  type="file" 
                  accept="application/pdf"
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
              </div>

              {file && (
                <div className="flex items-center gap-3 bg-gray-50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                  <File className="text-gray-400" size={20} />
                  <div className="flex-1 truncate">
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{file.name}</p>
                    <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
              )}

              <div className="pt-2">
                <button
                  onClick={handleUpload}
                  disabled={!file}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Start Upload & Ingestion
                </button>
              </div>
            </div>
          ) : (
            <div className="py-8 flex flex-col items-center justify-center space-y-6">
              <div className="relative">
                <Loader2 className="animate-spin text-blue-600" size={48} />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
                    <span className="text-[10px] font-bold text-blue-700 dark:text-blue-300">AI</span>
                  </div>
                </div>
              </div>
              <div className="text-center space-y-2">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">Processing Document</h3>
                <p className="text-sm text-blue-600 dark:text-blue-400 font-medium animate-pulse">
                  {progressStage}
                </p>
                <p className="text-xs text-gray-500 px-4">
                  This multi-pass pipeline extracts tables via vision models and indexes chunks via Gemini embeddings.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
