import { Routes, Route } from "react-router-dom";
import SetupScreen from "./components/SetupScreen";
import DebateView from "./components/DebateView";
import ComparisonView from "./components/ComparisonView";
import HistoryPage from "./components/HistoryPage";

export default function App() {
  return (
    <div className="h-screen flex flex-col">
      <div
        role="note"
        className="shrink-0 bg-amber-100 border-b border-amber-300 text-amber-900 text-xs sm:text-sm text-center px-4 py-1.5"
      >
        <strong>AI-generated simulation, not actual party statements.</strong>{" "}
        All party contributions are produced by language models for research
        purposes.
      </div>
      <div className="flex-1 min-h-0 overflow-auto">
        <Routes>
          <Route path="/" element={<SetupScreen />} />
          <Route path="/debate/:debateId" element={<DebateView />} />
          <Route path="/compare" element={<ComparisonView />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </div>
    </div>
  );
}
