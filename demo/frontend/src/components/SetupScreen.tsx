import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getConditions, getTopics, startDebate } from "../api";
import type { ConditionConfig, TopicConfig } from "../types";
import { TOPIC_LABELS } from "../constants";
import { Columns2, History, Play } from "lucide-react";

export default function SetupScreen() {
  const navigate = useNavigate();
  const [topics, setTopics] = useState<TopicConfig[]>([]);
  const [conditions, setConditions] = useState<ConditionConfig[]>([]);
  const [selectedTopic, setSelectedTopic] = useState("");
  const [selectedCondition, setSelectedCondition] = useState("");
  const [conditionTab, setConditionTab] = useState<"baseline" | "strategy">("baseline");
  const [numRounds, setNumRounds] = useState(10);
  const [language, setLanguage] = useState<"de" | "en">("de");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getTopics().then(setTopics).catch(console.error);
    getConditions().then(setConditions).catch(console.error);
  }, []);

  const baselines = conditions.filter((c) => c.type === "baseline");
  const strategies = conditions.filter((c) => c.type === "strategy");

  const handleStart = async () => {
    if (!selectedTopic || !selectedCondition) {
      setError("Please select both a topic and a condition.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const info = await startDebate({
        topic_id: selectedTopic,
        condition_id: selectedCondition,
        num_rounds: numRounds,
        language,
      });
      navigate(`/debate/${info.debate_id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start debate");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12">
      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          AI Debate Hub
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">
          AI-Moderated Multi-Agent Political Debate
        </p>
      </div>

      {/* Setup card */}
      <div className="w-full max-w-2xl card p-8 space-y-8">
        {/* Topic Selection */}
        <section>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Debate Topic
          </h2>
          <div className="grid grid-cols-2 gap-3">
            {topics.map((t) => {
              const label = TOPIC_LABELS[t.id];
              return (
                <button
                  key={t.id}
                  onClick={() => setSelectedTopic(t.id)}
                  className={`p-4 rounded-lg border text-left transition-all ${
                    selectedTopic === t.id
                      ? "border-gray-900 dark:border-white bg-gray-50 dark:bg-gray-700 ring-1 ring-gray-900 dark:ring-white"
                      : "border-gray-200 dark:border-gray-600 hover:border-gray-400"
                  }`}
                >
                  <span className="font-medium text-sm">
                    {label?.title ?? t.title}
                  </span>
                  <span className="block text-xs text-gray-500 mt-0.5">
                    {label?.subtitle ?? ""}
                  </span>
                  <span
                    className={`inline-block mt-2 text-xs px-2 py-0.5 rounded-full ${
                      t.type === "empirical"
                        ? "bg-blue-50 text-blue-700"
                        : "bg-purple-50 text-purple-700"
                    }`}
                  >
                    {t.type}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        {/* Condition Selection — mutually exclusive tabs */}
        <section>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Experimental Condition
          </h2>
          <p className="text-xs text-gray-400 mb-4">
            Choose <strong>one</strong> condition. Baselines and strategies are independent — pick one or the other.
          </p>

          {/* Tab bar */}
          <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
            <button
              onClick={() => { setConditionTab("baseline"); setSelectedCondition(""); }}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                conditionTab === "baseline"
                  ? "border-gray-900 dark:border-white text-gray-900 dark:text-white"
                  : "border-transparent text-gray-400 hover:text-gray-600"
              }`}
            >
              Baselines ({baselines.length})
            </button>
            <button
              onClick={() => { setConditionTab("strategy"); setSelectedCondition(""); }}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                conditionTab === "strategy"
                  ? "border-gray-900 dark:border-white text-gray-900 dark:text-white"
                  : "border-transparent text-gray-400 hover:text-gray-600"
              }`}
            >
              Strategies ({strategies.length})
            </button>
          </div>

          {/* Condition cards */}
          <div className="grid grid-cols-2 gap-2">
            {(conditionTab === "baseline" ? baselines : strategies).map((c) => (
              <button
                key={c.id}
                onClick={() => setSelectedCondition(c.id)}
                className={`p-3 rounded-lg border text-left transition-all ${
                  selectedCondition === c.id
                    ? conditionTab === "baseline"
                      ? "border-gray-900 dark:border-white bg-gray-50 dark:bg-gray-700 ring-1 ring-gray-900 dark:ring-white"
                      : "border-blue-600 bg-blue-50 dark:bg-blue-900/30 ring-1 ring-blue-600"
                    : "border-gray-200 dark:border-gray-600 hover:border-gray-400"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm">{c.label}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
                    conditionTab === "baseline"
                      ? "bg-gray-100 text-gray-500"
                      : "bg-blue-100 text-blue-700"
                  }`}>
                    {conditionTab === "baseline" ? "BASELINE" : "STRATEGY"}
                  </span>
                </div>
                <span className="block text-xs text-gray-500 mt-0.5">
                  {c.description || c.id.replace(/_/g, " ")}
                </span>
              </button>
            ))}
          </div>

          {/* Selection indicator */}
          {selectedCondition && (
            <div className="mt-3 text-xs text-gray-500">
              Selected: <span className="font-medium text-gray-900 dark:text-white">{selectedCondition.replace(/_/g, " ")}</span>
              {" "}({conditionTab})
            </div>
          )}
        </section>

        {/* Settings row */}
        <section className="flex items-center gap-6">
          {/* Rounds slider */}
          <div className="flex-1">
            <label className="text-sm text-gray-600 dark:text-gray-400">
              Rounds: <span className="font-medium text-gray-900 dark:text-white">{numRounds}</span>
            </label>
            <input
              type="range"
              min={1}
              max={10}
              value={numRounds}
              onChange={(e) => setNumRounds(Number(e.target.value))}
              className="w-full mt-1"
            />
          </div>

          {/* Language toggle */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Language:</span>
            <button
              onClick={() => setLanguage(language === "de" ? "en" : "de")}
              className="px-3 py-1 rounded border border-gray-300 text-sm font-medium"
            >
              {language === "de" ? "🇩🇪 DE" : "🇬🇧 EN"}
            </button>
          </div>
        </section>

        {/* Error */}
        {error && (
          <p className="text-sm text-red-600">{error}</p>
        )}

        {/* Actions */}
        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleStart}
            disabled={loading}
            className="btn-primary flex items-center gap-2 px-6 py-3 text-base"
          >
            <Play size={18} />
            {loading ? "Initializing..." : "Start Debate"}
          </button>
          <button
            onClick={() => navigate("/compare")}
            className="btn-outline flex items-center gap-2"
          >
            <Columns2 size={16} />
            Split View
          </button>
          <button
            onClick={() => navigate("/history")}
            className="btn-outline flex items-center gap-2"
          >
            <History size={16} />
            History
          </button>
        </div>
      </div>
    </div>
  );
}
