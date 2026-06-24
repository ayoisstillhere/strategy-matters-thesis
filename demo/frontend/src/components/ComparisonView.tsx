import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getConditions, getTopics, startDebate } from "../api";
import { useDebate } from "../hooks/useDebate";
import TurnCard from "./TurnCard";
import InterventionCard from "./InterventionCard";
import type { ConditionConfig, TopicConfig } from "../types";
import { TOPIC_LABELS } from "../constants";
import { ArrowLeft, Play } from "lucide-react";

export default function ComparisonView() {
  const navigate = useNavigate();
  const [topics, setTopics] = useState<TopicConfig[]>([]);
  const [conditions, setConditions] = useState<ConditionConfig[]>([]);
  const [selectedTopic, setSelectedTopic] = useState("");
  const [conditionA, setConditionA] = useState("");
  const [conditionB, setConditionB] = useState("");
  const [debateIdA, setDebateIdA] = useState<string | null>(null);
  const [debateIdB, setDebateIdB] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const debateA = useDebate(debateIdA);
  const debateB = useDebate(debateIdB);

  useEffect(() => {
    getTopics().then(setTopics).catch(console.error);
    getConditions().then(setConditions).catch(console.error);
  }, []);

  const handleStartBoth = async () => {
    if (!selectedTopic || !conditionA || !conditionB) return;
    setLoading(true);
    try {
      const [a, b] = await Promise.all([
        startDebate({ topic_id: selectedTopic, condition_id: conditionA, num_rounds: 5, language: "de" }),
        startDebate({ topic_id: selectedTopic, condition_id: conditionB, num_rounds: 5, language: "de" }),
      ]);
      setDebateIdA(a.debate_id);
      setDebateIdB(b.debate_id);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const started = debateIdA !== null && debateIdB !== null;

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="flex items-center gap-4">
          <button onClick={() => navigate("/")} className="text-gray-500 hover:text-gray-900">
            <ArrowLeft size={20} />
          </button>
          <h1 className="font-semibold text-gray-900 dark:text-white">
            Side-by-Side Comparison
          </h1>
        </div>
        {!started && (
          <div className="flex items-center gap-3">
            <select
              value={selectedTopic}
              onChange={(e) => setSelectedTopic(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="">Select topic...</option>
              {topics.map((t) => (
                <option key={t.id} value={t.id}>
                  {TOPIC_LABELS[t.id]?.title ?? t.title}
                </option>
              ))}
            </select>
            <select
              value={conditionA}
              onChange={(e) => setConditionA(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="">Condition A...</option>
              {conditions.map((c) => (
                <option key={c.id} value={c.id}>{c.label} ({c.id})</option>
              ))}
            </select>
            <select
              value={conditionB}
              onChange={(e) => setConditionB(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="">Condition B...</option>
              {conditions.map((c) => (
                <option key={c.id} value={c.id}>{c.label} ({c.id})</option>
              ))}
            </select>
            <button
              onClick={handleStartBoth}
              disabled={loading || !selectedTopic || !conditionA || !conditionB}
              className="btn-primary flex items-center gap-1 text-sm"
            >
              <Play size={14} /> Start Both
            </button>
          </div>
        )}
      </header>

      {/* Split panels */}
      {started ? (
        <div className="flex-1 flex overflow-hidden">
          {/* Panel A */}
          <div className="flex-1 overflow-y-auto border-r border-gray-200 dark:border-gray-700 px-4 py-4">
            <div className="text-center mb-4">
              <span className="text-xs font-medium bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                A: {debateA.info?.condition_label ?? conditionA}
              </span>
            </div>
            <div className="space-y-3 max-w-xl mx-auto">
              {debateA.timeline.map((item) => {
                if (item.type === "round_start") {
                  return (
                    <div key={`a-r-${item.roundNumber}`} className="flex items-center gap-2 my-4">
                      <div className="flex-1 h-px bg-gray-200" />
                      <span className="text-xs text-gray-400">Round {item.roundNumber}</span>
                      <div className="flex-1 h-px bg-gray-200" />
                    </div>
                  );
                }
                if (item.type === "turn") return <TurnCard key={item.data.turn_id} turn={item.data} />;
                if (item.type === "intervention") return <InterventionCard key={item.data.intervention_id} intervention={item.data} />;
                return null;
              })}
            </div>
          </div>

          {/* Panel B */}
          <div className="flex-1 overflow-y-auto px-4 py-4">
            <div className="text-center mb-4">
              <span className="text-xs font-medium bg-green-100 text-green-700 px-2 py-1 rounded-full">
                B: {debateB.info?.condition_label ?? conditionB}
              </span>
            </div>
            <div className="space-y-3 max-w-xl mx-auto">
              {debateB.timeline.map((item) => {
                if (item.type === "round_start") {
                  return (
                    <div key={`b-r-${item.roundNumber}`} className="flex items-center gap-2 my-4">
                      <div className="flex-1 h-px bg-gray-200" />
                      <span className="text-xs text-gray-400">Round {item.roundNumber}</span>
                      <div className="flex-1 h-px bg-gray-200" />
                    </div>
                  );
                }
                if (item.type === "turn") return <TurnCard key={item.data.turn_id} turn={item.data} />;
                if (item.type === "intervention") return <InterventionCard key={item.data.intervention_id} intervention={item.data} />;
                return null;
              })}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-gray-400">
          <p>Select a topic and two conditions, then click "Start Both" to compare.</p>
        </div>
      )}
    </div>
  );
}
