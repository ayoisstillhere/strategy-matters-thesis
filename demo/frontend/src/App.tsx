import { Routes, Route } from "react-router-dom";
import SetupScreen from "./components/SetupScreen";
import DebateView from "./components/DebateView";
import ComparisonView from "./components/ComparisonView";
import HistoryPage from "./components/HistoryPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<SetupScreen />} />
      <Route path="/debate/:debateId" element={<DebateView />} />
      <Route path="/compare" element={<ComparisonView />} />
      <Route path="/history" element={<HistoryPage />} />
    </Routes>
  );
}
