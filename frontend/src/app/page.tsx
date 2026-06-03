"use client";

import { useState, useEffect } from "react";
import { Chat } from "@/components/chat";
import { Moon, Sun } from "lucide-react";

export default function Home() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);

  return (
    <main className="flex h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <div className="flex-1 flex flex-col max-w-5xl mx-auto bg-white dark:bg-gray-950 shadow-xl border-x border-gray-200 dark:border-gray-800">
        <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex justify-between items-center bg-white dark:bg-gray-950">
          <div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">FiscalAgent</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">Advanced Financial RAG Agent</p>
          </div>
          <button 
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-300 transition-colors"
            title="Toggle Theme"
          >
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
        <div className="flex-1 overflow-hidden relative">
          <Chat />
        </div>
      </div>
    </main>
  );
}
