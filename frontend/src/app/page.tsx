import { Chat } from "@/components/chat";

export default function Home() {
  return (
    <main className="flex h-screen bg-white">
      {/* Left side: PDF Viewer Placeholder */}
      <div className="flex-1 border-r bg-gray-50 p-8 flex flex-col relative">
        <h2 className="text-xl font-bold mb-4 text-black">Document Viewer</h2>
        <div className="flex-1 bg-white border shadow-sm relative overflow-hidden flex items-center justify-center text-gray-400">
          <p>PDF will be rendered here.</p>
          
          {/* Example of how a bounding box would be rendered over the PDF based on [ymin, xmin, ymax, xmax] */}
          {/* Using normalized coordinates (0-1000) converted to percentages for CSS top, left, width, height */}
          {/* Example coordinates: [100, 50, 150, 300] -> ymin=10%, xmin=5%, ymax=15%, xmax=30% */}
          {/* <div 
            className="absolute border-2 border-red-500 bg-yellow-200/30 transition-all duration-300 pointer-events-none"
            style={{ 
              top: '10%', 
              left: '5%', 
              height: '5%', // 15% - 10%
              width: '25%'  // 30% - 5%
            }}
          /> */}
        </div>
      </div>

      {/* Right side: Agent Chat Interface */}
      <div className="w-[450px] flex flex-col bg-white">
        <div className="p-4 border-b bg-gray-50">
          <h1 className="text-lg font-bold text-black">Financial RAG Agent</h1>
          <p className="text-xs text-gray-500">Gemini 1.5 Pro | RRF Hybrid Search</p>
        </div>
        <div className="flex-1 overflow-hidden">
          <Chat />
        </div>
      </div>
    </main>
  );
}
