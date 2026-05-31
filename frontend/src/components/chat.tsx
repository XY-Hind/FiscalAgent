"use client";

import { useState } from "react";

interface Citation {
  exact_quote: string;
  bounding_box: number[];
  page_number?: number;
}

interface AnswerWithEvidence {
  answer: string;
  reasoning: string;
  citations: Citation[];
}

export function Chat() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<{role: "user" | "assistant", content: string, citations?: Citation[]}[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = query;
    setQuery("");
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setIsStreaming(true);

    try {
      // Connect to FastAPI SSE Endpoint
      // Bypassing Next.js cache using `cache: "no-store"`
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
              // Handle structured final output
              if (parsed.answer && parsed.citations) {
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = {
                    role: "assistant",
                    content: parsed.answer,
                    citations: parsed.citations
                  };
                  return newMessages;
                });
              } else {
                // Handle raw updates (mocking streaming)
                currentAssistantMessage += " " + dataStr;
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = {
                    role: "assistant",
                    content: currentAssistantMessage,
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
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`p-3 rounded-lg max-w-[80%] ${msg.role === "user" ? "bg-blue-600 text-white self-end ml-auto" : "bg-gray-100 text-black self-start"}`}>
            <p>{msg.content}</p>
            {msg.citations && msg.citations.length > 0 && (
              <div className="mt-2 p-2 bg-yellow-100 rounded text-sm text-yellow-900 border border-yellow-300">
                <strong>Citations:</strong>
                <ul className="list-disc pl-4 mt-1">
                  {msg.citations.map((cit, cIdx) => (
                    <li key={cIdx}>
                      "{cit.exact_quote}" (Box: {cit.bounding_box.join(", ")})
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {isStreaming && <div className="text-gray-500 text-sm">Agent is thinking...</div>}
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t flex gap-2">
        <input 
          type="text" 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about the financial reports..."
          className="flex-1 border p-2 rounded text-black"
          disabled={isStreaming}
        />
        <button 
          type="submit" 
          disabled={isStreaming}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
