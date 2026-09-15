import { useState } from "react";
import { FileText, Upload, ChevronDown, Info, Lock } from "lucide-react";

interface DocumentSelectorProps {
  documents: any[];
  activeDocumentId: string | null;
  setActiveDocumentId: (id: string) => void;
  onOpenUpload: () => void;
  isLocked: boolean;
}

export function DocumentSelector({ 
  documents, 
  activeDocumentId, 
  setActiveDocumentId, 
  onOpenUpload, 
  isLocked 
}: DocumentSelectorProps) {
  const [mode, setMode] = useState<"demo" | "upload">("demo");

  if (isLocked) {
    const activeDoc = documents.find(d => d.id === activeDocumentId);
    const docName = activeDoc ? (activeDoc.title || activeDoc.file_name) : "All Documents";

    return (
      <div className="flex items-center gap-2 bg-blue-50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 px-4 py-2 rounded-xl text-xs font-medium border border-blue-200 dark:border-blue-800/50">
        <Lock size={14} className="shrink-0" />
        <span>Document context locked for this chat session: <strong>{docName}</strong>. Start a New Chat to switch documents.</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Mode Toggle */}
      <div className="flex bg-gray-100 dark:bg-gray-800 p-1 rounded-lg w-fit">
        <button
          onClick={() => setMode("demo")}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
            mode === "demo" 
              ? "bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm" 
              : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          }`}
        >
          Demo Mode
          <div className="group relative">
            <Info size={14} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
            <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 hidden group-hover:block w-48 p-2 bg-gray-900 text-white text-[10px] rounded-md shadow-lg z-10 text-center">
              Instant response mode using pre-loaded annual reports.
            </div>
          </div>
        </button>
        <button
          onClick={() => setMode("upload")}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
            mode === "upload" 
              ? "bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm" 
              : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          }`}
        >
          Upload Custom PDF
          <div className="group relative">
            <Info size={14} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
            <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 hidden group-hover:block w-48 p-2 bg-gray-900 text-white text-[10px] rounded-md shadow-lg z-10 text-center">
              Upload a corporate annual report to chat with.
            </div>
          </div>
        </button>
      </div>

      {/* Mode Content */}
      <div className="flex items-center gap-3">
        {mode === "demo" ? (
          <div className="relative flex items-center bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 shadow-sm hover:border-gray-300 dark:hover:border-gray-600 transition-colors group min-w-[250px]">
            <FileText size={16} className="text-gray-500 dark:text-gray-400 mr-2" />
            <select 
              value={activeDocumentId || ""}
              onChange={(e) => setActiveDocumentId(e.target.value)}
              className="w-full bg-transparent border-none outline-none text-sm font-medium text-gray-700 dark:text-gray-200 cursor-pointer appearance-none pr-6 focus:ring-0"
            >
              <option value="">Search all documents</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.title || doc.file_name} {doc.fiscal_year ? `(${doc.fiscal_year})` : ""}
                </option>
              ))}
            </select>
            <ChevronDown size={14} className="text-gray-400 absolute right-3 pointer-events-none group-hover:text-gray-500" />
          </div>
        ) : (
          <button 
            onClick={onOpenUpload}
            className="flex items-center justify-center w-full sm:w-auto gap-2 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800/50 hover:bg-blue-100 dark:hover:bg-blue-900/40 rounded-lg px-4 py-2 text-sm font-medium transition-colors shadow-sm"
          >
            <Upload size={16} />
            Click here to upload your PDF
          </button>
        )}
      </div>
    </div>
  );
}
