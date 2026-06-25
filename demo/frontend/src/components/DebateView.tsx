import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useDebate } from "../hooks/useDebate";
import { injectIntervention } from "../api";
import TurnCard from "./TurnCard";
import InterventionCard from "./InterventionCard";
import ScorePanel from "./ScorePanel";
import { TOPIC_LABELS } from "../constants";
import { ArrowLeft, Send, Wifi, WifiOff } from "lucide-react";

export default function DebateView() {
  const { debateId } = useParams<{ debateId: string }>();
  const navigate = useNavigate();
  const { info, timeline, roundSummaries, debateConfig, connected } = useDebate(debateId ?? null);
  const [injectionText, setInjectionText] = useState("");
  const [injecting, setInjecting] = useState(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to bottom when new items arrive
  useEffect(() => {
    if (autoScroll && transcriptEndRef.current) {
      transcriptEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [timeline.length, autoScroll]);

  const handleInject = async () => {
    if (!debateId || !injectionText.trim()) return;
    setInjecting(true);
    try {
      await injectIntervention(debateId, injectionText.trim());
      setInjectionText("");
    } catch (e) {
      console.error("Injection failed:", e);
    } finally {
      setInjecting(false);
    }
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    setAutoScroll(atBottom);
  };

  const topicLabel = info ? TOPIC_LABELS[info.topic_id] : null;

  return (
    <div className="h-screen flex flex-col">
      {/* Header Bar */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate("/")}
            className="text-gray-500 hover:text-gray-900 dark:hover:text-white"
          >
            <ArrowLeft size={20} />
          </button>
          <h1 className="font-semibold text-gray-900 dark:text-white">
            AI Debate Hub
          </h1>
          {topicLabel && (
            <span className="text-sm text-gray-500">
              Topic: {topicLabel.title}
            </span>
          )}
          {info && (
            <span className="text-sm text-gray-500">
              Strategy: {info.condition_label}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Connection status */}
          <span className="flex items-center gap-1 text-xs text-gray-500">
            {connected ? (
              <><Wifi size={14} className="text-green-500" /> Connected</>
            ) : (
              <><WifiOff size={14} className="text-red-500" /> Disconnected</>
            )}
          </span>
          {/* Status badge */}
          {info && (
            <span
              className={`text-xs px-2 py-1 rounded-full font-medium ${
                info.status === "running"
                  ? "bg-green-100 text-green-700 animate-pulse"
                  : info.status === "completed"
                  ? "bg-gray-100 text-gray-700"
                  : info.status === "error"
                  ? "bg-red-100 text-red-700"
                  : "bg-yellow-100 text-yellow-700"
              }`}
            >
              {info.status.toUpperCase()}
            </span>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Transcript area */}
        <main
          className="flex-1 overflow-y-auto px-6 py-4"
          onScroll={handleScroll}
        >
          {/* Session header */}
          {info && (
            <div className="text-center mb-6">
              <span className="inline-block text-xs px-3 py-1 rounded-full bg-blue-50 text-blue-700 font-medium uppercase tracking-wide">
                Active Session: {topicLabel?.title ?? info.topic_id}
              </span>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mt-2">
                Interactive Debate Transcript
              </h2>
            </div>
          )}

          {/* Topic framing card — shows what agents were actually told */}
          {(topicLabel || debateConfig) && (
            <div className="max-w-3xl mx-auto mb-6 p-5 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-bold text-indigo-700 dark:text-indigo-300 uppercase tracking-wide">
                  Debate Framing
                </span>
                {info && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-100 dark:bg-indigo-800 text-indigo-600 dark:text-indigo-300 font-medium">
                    {info.condition_label}
                  </span>
                )}
              </div>
              {topicLabel && (
                <h3 className="text-base font-semibold text-indigo-900 dark:text-indigo-100 mb-1">
                  {topicLabel.title}
                  {topicLabel.subtitle && (
                    <span className="text-sm font-normal text-indigo-600 dark:text-indigo-300 ml-2">
                      ({topicLabel.subtitle})
                    </span>
                  )}
                </h3>
              )}
              {debateConfig?.framing_prompt ? (
                <p className="text-sm text-indigo-800 dark:text-indigo-200 leading-relaxed mt-2">
                  {String(debateConfig.framing_prompt)}
                </p>
              ) : null}
              {debateConfig?.language ? (
                <p className="text-[10px] text-indigo-500 mt-2">
                  Language: {String(debateConfig.language) === "de" ? "Deutsch" : "English"}
                </p>
              ) : null}
            </div>
          )}

          {/* Timeline */}
          <div className="max-w-3xl mx-auto space-y-4">
            {timeline.map((item, idx) => {
              if (item.type === "round_start") {
                return (
                  <div
                    key={`round-${item.roundNumber}`}
                    className="flex items-center gap-3 my-6"
                  >
                    <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700" />
                    <span className="text-xs font-medium text-gray-400 uppercase">
                      Round {item.roundNumber}
                    </span>
                    <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700" />
                  </div>
                );
              }
              if (item.type === "turn") {
                return <TurnCard key={item.data.turn_id} turn={item.data} />;
              }
              if (item.type === "intervention") {
                return (
                  <InterventionCard
                    key={item.data.intervention_id}
                    intervention={item.data}
                  />
                );
              }
              return null;
            })}

            {/* Streaming indicator */}
            {info?.status === "running" && (
              <div className="flex items-center gap-2 text-sm text-gray-400 py-2">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
                <span>Generating next turn...</span>
              </div>
            )}

            <div ref={transcriptEndRef} />
          </div>
        </main>

        {/* Right sidebar — Score Panel */}
        <aside className="w-72 border-l border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-y-auto p-4 hidden lg:block">
          <ScorePanel info={info} roundSummaries={roundSummaries} />
        </aside>
      </div>

      {/* Bottom injection bar */}
      <footer className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-6 py-3">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <input
            type="text"
            value={injectionText}
            onChange={(e) => setInjectionText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleInject()}
            placeholder="Inject manual moderator intervention..."
            disabled={info?.status !== "running"}
            className="flex-1 px-4 py-2.5 rounded-lg border border-gray-300 dark:border-gray-600 
                       bg-gray-50 dark:bg-gray-700 text-sm
                       focus:outline-none focus:ring-2 focus:ring-gray-400
                       disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleInject}
            disabled={injecting || !injectionText.trim() || info?.status !== "running"}
            className="btn-primary flex items-center gap-2 px-5 py-2.5"
          >
            Send <Send size={16} />
          </button>
        </div>
      </footer>
    </div>
  );
}
