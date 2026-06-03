"use client";

import { useState } from "react";
import { Send, FileText } from "lucide-react";

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

export function Chat() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = query;
    setQuery("");
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setIsStreaming(true);

    try {
      const response = await fetch("http://localhost:8000/api/v1/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: "test-session-123",
          message: userMessage,
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
    <div className="flex flex-col h-full bg-white dark:bg-gray-950 transition-colors">
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>
            <div className={`p-4 rounded-xl max-w-[85%] ${
              msg.role === "user" 
                ? "bg-blue-600 text-white" 
                : "bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-800"
            }`}>
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
            
            {msg.citations && msg.citations.length > 0 && (
              <div className="mt-3 w-full max-w-[85%] flex flex-wrap gap-2">
                {msg.citations.map((cit, cIdx) => (
                  <div key={cIdx} className="flex items-start gap-2 bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 w-full">
                    <FileText className="text-blue-500 mt-1 shrink-0" size={18} />
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {cit.section_name ? cit.section_name : "Document Section"}
                        {cit.page_number && <span className="text-gray-500 dark:text-gray-400 font-normal ml-2">Page {cit.page_number}</span>}
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mt-1 italic">"{cit.exact_quote}"</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {isStreaming && (
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-sm">
            <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            Agent is thinking...
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
        <div className="relative flex items-center">
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about the financial reports..."
            className="flex-1 border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white p-3 pr-12 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
            disabled={isStreaming}
          />
          <button 
            type="submit" 
            disabled={isStreaming || !query.trim()}
            className="absolute right-2 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  );
}
