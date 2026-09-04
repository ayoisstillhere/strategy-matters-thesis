import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getConditions, getTopics, startDebate, injectIntervention } from "../api";
import { useDebate } from "../hooks/useDebate";
import TurnCard from "./TurnCard";
import InterventionCard from "./InterventionCard";
// ScorePanel used inline via MiniProgress
import type { ConditionConfig, TopicConfig } from "../types";
import { TOPIC_LABELS, DIMENSIONS, getScoreBarColor } from "../constants";
import { ArrowLeft, Play, Send } from "lucide-react";
import ScoreChart from "./ScoreChart";

function MiniProgress({ info, roundSummaries }: { info: any; roundSummaries: any[] }) {
  const progress = info
    ? Math.round((info.current_round / info.total_rounds) * 100)
    : 0;
  const latestSummary = roundSummaries[roundSummaries.length - 1] ?? null;

  return (
    <div className="mb-4 px-2 space-y-2">
      {/* Progress bar */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>Round {info?.current_round ?? 0}/{info?.total_rounds ?? "?"}</span>
        <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
          info?.status === "running" ? "bg-green-100 text-green-700" :
          info?.status === "completed" ? "bg-gray-100 text-gray-600" : "bg-yellow-100 text-yellow-700"
        }`}>
          {info?.status?.toUpperCase() ?? "—"}
        </span>
      </div>
      <div className="w-full h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${progress}%` }} />
      </div>
      {/* Mini scores */}
      {latestSummary?.scores && (
        <div className="grid grid-cols-2 gap-1 mt-2">
          {DIMENSIONS.map((dim) => {
            const score = latestSummary.scores[dim.key] ?? 0;
            return (
              <div key={dim.key} className="flex items-center gap-1">
                <span className="text-[10px] text-gray-500 w-8 truncate">{dim.abbrev}</span>
                <div className="flex-1 h-1 bg-gray-200 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${getScoreBarColor(score)}`} style={{ width: `${(score/5)*100}%` }} />
                </div>
                <span className="text-[10px] text-gray-600">{score > 0 ? score.toFixed(1) : "—"}</span>
              </div>
            );
          })}
        </div>
      )}
      {info && (
        <div className="text-[10px] text-gray-400">
          Interventions: {info.intervention_count ?? 0}
        </div>
      )}
    </div>
  );
}

export default function ComparisonView() {
  const navigate = useNavigate();
  const [topics, setTopics] = useState<TopicConfig[]>([]);
  const [conditions, setConditions] = useState<ConditionConfig[]>([]);
  const [selectedTopic, setSelectedTopic] = useState("");
  const [conditionA, setConditionA] = useState("");
  const [conditionB, setConditionB] = useState("");
  const [numRounds, setNumRounds] = useState(5);
  const [language, setLanguage] = useState<"de" | "en">("de");
  const [debateIdA, setDebateIdA] = useState<string | null>(null);
  const [debateIdB, setDebateIdB] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [injectionA, setInjectionA] = useState("");
  const [injectionB, setInjectionB] = useState("");
  const [injectingA, setInjectingA] = useState(false);
  const [injectingB, setInjectingB] = useState(false);

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
        startDebate({ topic_id: selectedTopic, condition_id: conditionA, num_rounds: numRounds, language }),
        startDebate({ topic_id: selectedTopic, condition_id: conditionB, num_rounds: numRounds, language }),
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
  const topicLabel = TOPIC_LABELS[selectedTopic];

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
          <div className="flex flex-wrap items-center gap-2">
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
              <optgroup label="Baselines">
                {conditions.filter(c => c.type === "baseline").map((c) => (
                  <option key={c.id} value={c.id}>{c.label}</option>
                ))}
              </optgroup>
              <optgroup label="Strategies">
                {conditions.filter(c => c.type === "strategy").map((c) => (
                  <option key={c.id} value={c.id}>{c.label}</option>
                ))}
              </optgroup>
            </select>
            <select
              value={conditionB}
              onChange={(e) => setConditionB(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="">Condition B...</option>
              <optgroup label="Baselines">
                {conditions.filter(c => c.type === "baseline").map((c) => (
                  <option key={c.id} value={c.id}>{c.label}</option>
                ))}
              </optgroup>
              <optgroup label="Strategies">
                {conditions.filter(c => c.type === "strategy").map((c) => (
                  <option key={c.id} value={c.id}>{c.label}</option>
                ))}
              </optgroup>
            </select>
            <select
              value={numRounds}
              onChange={(e) => setNumRounds(Number(e.target.value))}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              {[3, 5, 7, 10].map((n) => (
                <option key={n} value={n}>{n} rounds</option>
              ))}
            </select>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as "de" | "en")}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="de">Deutsch</option>
              <option value="en">English</option>
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
          <div className="flex-1 flex flex-col overflow-hidden border-r border-gray-200 dark:border-gray-700">
            <div className="text-center py-2 border-b border-gray-100">
              <span className="text-xs font-medium bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                A: {debateA.info?.condition_label ?? conditionA}
              </span>
            </div>
            {/* Injection bar A */}
            {debateA.info?.status === "running" && debateIdA && (
              <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-100 bg-gray-50">
                <input
                  type="text"
                  value={injectionA}
                  onChange={(e) => setInjectionA(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && injectionA.trim()) {
                      setInjectingA(true);
                      injectIntervention(debateIdA, injectionA.trim()).then(() => setInjectionA("")).finally(() => setInjectingA(false));
                    }
                  }}
                  placeholder="Inject intervention..."
                  className="flex-1 text-xs px-2 py-1 border border-gray-300 rounded"
                  disabled={injectingA}
                />
                <button
                  onClick={() => {
                    if (!injectionA.trim()) return;
                    setInjectingA(true);
                    injectIntervention(debateIdA, injectionA.trim()).then(() => setInjectionA("")).finally(() => setInjectingA(false));
                  }}
                  disabled={injectingA || !injectionA.trim()}
                  className="text-blue-600 hover:text-blue-800 disabled:text-gray-300"
                >
                  <Send size={14} />
                </button>
              </div>
            )}
            <MiniProgress info={debateA.info} roundSummaries={debateA.roundSummaries} />
            {debateA.roundSummaries.length > 0 && (
              <div className="px-3 pb-2">
                <ScoreChart roundSummaries={debateA.roundSummaries} height={160} compact />
              </div>
            )}
            <div className="flex-1 overflow-y-auto px-4 pb-4">
              {/* Topic framing */}
              {(topicLabel || debateA.debateConfig) && (
                <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <p className="text-xs font-semibold text-blue-700 dark:text-blue-300 uppercase mb-1">Debate Framing</p>
                  {topicLabel && <p className="text-sm font-medium text-blue-900 dark:text-blue-100">{topicLabel.title}</p>}
                  {debateA.debateConfig?.framing_prompt ? (
                    <p className="text-xs text-blue-800 dark:text-blue-200 mt-1 leading-relaxed">
                      {String(debateA.debateConfig.framing_prompt)}
                    </p>
                  ) : null}
                </div>
              )}
              <div className="space-y-3 max-w-xl mx-auto">
                {(debateA.info?.status === "pending" || debateA.info?.status === "running") && debateA.timeline.length === 0 && (
                  <div className="flex flex-col items-center justify-center py-10 text-center">
                    <div className="flex gap-1 mb-3">
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                    <p className="text-xs text-gray-500">Initializing models...</p>
                  </div>
                )}
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
          </div>

          {/* Panel B */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="text-center py-2 border-b border-gray-100">
              <span className="text-xs font-medium bg-green-100 text-green-700 px-2 py-1 rounded-full">
                B: {debateB.info?.condition_label ?? conditionB}
              </span>
            </div>
            {/* Injection bar B */}
            {debateB.info?.status === "running" && debateIdB && (
              <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-100 bg-gray-50">
                <input
                  type="text"
                  value={injectionB}
                  onChange={(e) => setInjectionB(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && injectionB.trim()) {
                      setInjectingB(true);
                      injectIntervention(debateIdB, injectionB.trim()).then(() => setInjectionB("")).finally(() => setInjectingB(false));
                    }
                  }}
                  placeholder="Inject intervention..."
                  className="flex-1 text-xs px-2 py-1 border border-gray-300 rounded"
                  disabled={injectingB}
                />
                <button
                  onClick={() => {
                    if (!injectionB.trim()) return;
                    setInjectingB(true);
                    injectIntervention(debateIdB, injectionB.trim()).then(() => setInjectionB("")).finally(() => setInjectingB(false));
                  }}
                  disabled={injectingB || !injectionB.trim()}
                  className="text-green-600 hover:text-green-800 disabled:text-gray-300"
                >
                  <Send size={14} />
                </button>
              </div>
            )}
            <MiniProgress info={debateB.info} roundSummaries={debateB.roundSummaries} />
            {debateB.roundSummaries.length > 0 && (
              <div className="px-3 pb-2">
                <ScoreChart roundSummaries={debateB.roundSummaries} height={160} compact />
              </div>
            )}
            <div className="flex-1 overflow-y-auto px-4 pb-4">
              {/* Topic framing */}
              {(topicLabel || debateB.debateConfig) && (
                <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <p className="text-xs font-semibold text-green-700 dark:text-green-300 uppercase mb-1">Debate Framing</p>
                  {topicLabel && <p className="text-sm font-medium text-green-900 dark:text-green-100">{topicLabel.title}</p>}
                  {debateB.debateConfig?.framing_prompt ? (
                    <p className="text-xs text-green-800 dark:text-green-200 mt-1 leading-relaxed">
                      {String(debateB.debateConfig.framing_prompt)}
                    </p>
                  ) : null}
                </div>
              )}
              <div className="space-y-3 max-w-xl mx-auto">
                {(debateB.info?.status === "pending" || debateB.info?.status === "running") && debateB.timeline.length === 0 && (
                  <div className="flex flex-col items-center justify-center py-10 text-center">
                    <div className="flex gap-1 mb-3">
                      <span className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                      <span className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                      <span className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                    <p className="text-xs text-gray-500">Initializing models...</p>
                  </div>
                )}
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
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-gray-400">
          <p>Select a topic and two conditions, then click "Start Both" to compare.</p>
        </div>
      )}
    </div>
  );
}
