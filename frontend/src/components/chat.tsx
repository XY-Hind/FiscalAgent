"use client";

import { useState, useRef, useEffect } from "react";
import { Send, FileText, Bot, PlusCircle } from "lucide-react";
import { DocumentSelector } from "@/components/document-selector";

interface Citation {
  exact_quote: string;
  bounding_box: number[];
  page_number?: number;
  section_name?: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

interface ChatProps {
  activeDocumentId: string | null;
  setActiveDocumentId: (id: string) => void;
  documents: any[];
  onOpenUpload: () => void;
}

export function Chat({ activeDocumentId, setActiveDocumentId, documents, onOpenUpload }: ChatProps) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Session State
  const [isLocked, setIsLocked] = useState(false);
  const [chatSessionId, setChatSessionId] = useState<string>(() => crypto.randomUUID());

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleNewChat = () => {
    setMessages([]);
    setChatSessionId(crypto.randomUUID());
    setIsLocked(false);
    setQuery("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = query;
    setQuery("");
    
    // Lock the session to the current document on the first message
    if (!isLocked) {
      setIsLocked(true);
    }

    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setIsStreaming(true);

    try {
      const response = await fetch("http://localhost:8000/api/v1/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: chatSessionId,
          message: userMessage,
          document_id: activeDocumentId,
        }),
        cache: "no-store",
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let currentAssistantMessage = "";
      setMessages(prev => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "").trim();
            if (dataStr === "[DONE]") {
              setIsStreaming(false);
              break;
            }
            
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.type === "token") {
                currentAssistantMessage += parsed.content;
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = {
                    ...newMessages[newMessages.length - 1],
                    content: currentAssistantMessage,
                  };
                  return newMessages;
                });
              } else if (parsed.type === "final") {
                const finalData = parsed.data;
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = {
                    role: "assistant",
                    content: finalData.answer,
                    citations: finalData.citations
                  };
                  return newMessages;
                });
              }
            } catch (e) {
              // Ignore parse errors from partial chunks
            }
          }
        }
      }
    } catch (error) {
      console.error("Error streaming chat:", error);
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-900 transition-colors">
      
      {/* Top Header inside chat for New Chat button */}
      <div className="flex justify-end px-6 pt-4">
        {messages.length > 0 && (
          <button 
            onClick={handleNewChat}
            className="flex items-center gap-1.5 text-xs font-medium text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <PlusCircle size={14} />
            New Chat
          </button>
        )}
      </div>

      {/* Scrollable Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="max-w-3xl mx-auto space-y-8 pb-10">
          
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center mt-10 text-gray-400 dark:text-gray-500">
              <Bot size={48} className="mb-4 opacity-50" />
              <p className="text-lg">How can I help with your financial research?</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center shrink-0 mt-1">
                  <Bot size={18} className="text-blue-600 dark:text-blue-300" />
                </div>
              )}
              
              <div className="flex flex-col gap-2 max-w-[80%]">
                <div className={`px-5 py-3.5 rounded-2xl ${
                  msg.role === "user" 
                    ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100" 
                    : "text-gray-900 dark:text-gray-100 prose dark:prose-invert prose-p:leading-relaxed"
                }`}>
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                </div>
                
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-2 flex flex-col gap-2">
                    {msg.citations.map((cit, cIdx) => (
                      <div key={cIdx} className="flex items-start gap-2 bg-blue-50 dark:bg-blue-900/10 p-3 rounded-xl border border-blue-100 dark:border-blue-900/30">
                        <FileText className="text-blue-500 mt-1 shrink-0" size={16} />
                        <div>
                          <div className="text-xs font-semibold text-blue-900 dark:text-blue-300">
                            {cit.section_name ? cit.section_name : "Document Section"}
                            {cit.page_number && <span className="opacity-70 ml-2">Page {cit.page_number}</span>}
                          </div>
                          <p className="text-sm text-gray-700 dark:text-gray-300 mt-1 italic">"{cit.exact_quote}"</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {isStreaming && (
            <div className="flex items-center gap-4 text-gray-500 dark:text-gray-400 mt-4">
              <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center shrink-0">
                <Bot size={18} className="text-blue-600 dark:text-blue-300" />
              </div>
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="shrink-0 bg-white dark:bg-gray-900 pb-6 px-4 pt-2">
        <div className="max-w-3xl mx-auto flex flex-col gap-4">
          
          <DocumentSelector 
            documents={documents}
            activeDocumentId={activeDocumentId}
            setActiveDocumentId={setActiveDocumentId}
            onOpenUpload={onOpenUpload}
            isLocked={isLocked}
          />

          {/* Form */}
          <form onSubmit={handleSubmit} className="relative mt-1">
            <div className="overflow-hidden rounded-2xl bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 focus-within:border-blue-500 dark:focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 shadow-sm transition-all">
              <input 
                type="text" 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Message FiscalAgent..."
                className="w-full bg-transparent text-gray-900 dark:text-white p-4 pr-14 text-sm focus:outline-none min-h-[56px]"
                disabled={isStreaming}
              />
              <button 
                type="submit" 
                disabled={isStreaming || !query.trim()}
                className="absolute right-3 top-1/2 -translate-y-1/2 w-9 h-9 flex items-center justify-center bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-40 disabled:hover:bg-blue-600 transition-colors shadow-sm"
              >
                <Send size={16} className={query.trim() && !isStreaming ? "ml-0.5" : ""} />
              </button>
            </div>
            <div className="text-center mt-2">
              <span className="text-[10px] text-gray-400 dark:text-gray-500">FiscalAgent can make mistakes. Verify important information.</span>
            </div>
          </form>

        </div>
      </div>
    </div>
  );
}
