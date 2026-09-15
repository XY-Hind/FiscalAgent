"use client";

import { useState, useEffect } from "react";
import { Chat } from "@/components/chat";
import { Moon, Sun } from "lucide-react";
import { UploadModal } from "@/components/upload-modal";

export default function Home() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<any[]>([]);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
  }, [isDarkMode]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/v1/documents");
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
        if (!activeDocumentId && data.documents.length > 0) {
          setActiveDocumentId(data.documents[0].id);
        }
      }
    } catch (e) {
      console.error("Failed to fetch documents", e);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleUploadComplete = (documentId: string) => {
    setIsUploadModalOpen(false);
    fetchDocuments().then(() => {
      setActiveDocumentId(documentId);
    });
  };

  return (
    <main className="flex flex-col h-screen w-full bg-white dark:bg-gray-900 transition-colors duration-200">
      
      {/* Sleek Minimal Header */}
      <div className="flex justify-between items-center p-4 h-14 shrink-0 bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800/50">
        <div className="flex items-center gap-2">
          <span className="bg-blue-600 text-white p-1 rounded-md text-sm font-bold shadow-sm">FA</span>
          <h1 className="text-lg font-bold text-gray-900 dark:text-gray-100">FiscalAgent</h1>
        </div>
        
        <button 
          onClick={() => setIsDarkMode(!isDarkMode)}
          className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 transition-colors"
          title="Toggle Theme"
        >
          {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
      
      {/* Chat Area taking full height, centering content internally */}
      <div className="flex-1 overflow-hidden relative">
        <Chat 
          activeDocumentId={activeDocumentId} 
          setActiveDocumentId={setActiveDocumentId}
          documents={documents}
          onOpenUpload={() => setIsUploadModalOpen(true)}
        />
      </div>

      {isUploadModalOpen && (
        <UploadModal 
          onClose={() => setIsUploadModalOpen(false)} 
          onUploadComplete={handleUploadComplete} 
        />
      )}
    </main>
  );
}
