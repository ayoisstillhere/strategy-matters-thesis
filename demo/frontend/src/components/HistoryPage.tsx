import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listDebates } from "../api";
import type { DebateInfo } from "../types";
import { TOPIC_LABELS } from "../constants";
import {
  ArrowLeft,
  Clock,
  MessageSquare,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  History,
} from "lucide-react";

function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-green-50 text-green-700 border border-green-200">
          <CheckCircle2 size={12} />
          Completed
        </span>
      );
    case "running":
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
          <Loader2 size={12} className="animate-spin" />
          Running
        </span>
      );
    case "error":
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200">
          <AlertTriangle size={12} />
          Error
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-gray-50 text-gray-600 border border-gray-200">
          <Clock size={12} />
          Pending
        </span>
      );
  }
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HistoryPage() {
  const navigate = useNavigate();
  const [debates, setDebates] = useState<DebateInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    listDebates()
      .then((data) => {
        // Sort by created_at descending (most recent first)
        const sorted = [...data].sort((a, b) => {
          const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
          const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
          return tb - ta;
        });
        setDebates(sorted);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="flex items-center gap-4 px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <button
          onClick={() => navigate("/")}
          className="text-gray-500 hover:text-gray-900 dark:hover:text-white"
        >
          <ArrowLeft size={20} />
        </button>
        <div className="flex items-center gap-2">
          <History size={20} className="text-gray-600 dark:text-gray-300" />
          <h1 className="font-semibold text-gray-900 dark:text-white text-lg">
            Debate History
          </h1>
        </div>
        <span className="text-sm text-gray-400 ml-auto">
          {debates.length} debate{debates.length !== 1 ? "s" : ""}
        </span>
      </header>

      {/* Content */}
      <main className="flex-1 p-6 max-w-5xl mx-auto w-full">
        {loading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 size={32} className="animate-spin text-gray-400" />
          </div>
        )}

        {error && (
          <div className="text-center py-12">
            <AlertTriangle size={32} className="mx-auto text-red-400 mb-3" />
            <p className="text-red-600">{error}</p>
          </div>
        )}

        {!loading && !error && debates.length === 0 && (
          <div className="text-center py-20">
            <MessageSquare
              size={48}
              className="mx-auto text-gray-300 dark:text-gray-600 mb-4"
            />
            <p className="text-gray-500 dark:text-gray-400 text-lg">
              No debates yet
            </p>
            <p className="text-gray-400 dark:text-gray-500 text-sm mt-1">
              Start a new debate from the setup screen
            </p>
            <button
              onClick={() => navigate("/")}
              className="btn-primary mt-6"
            >
              Start a Debate
            </button>
          </div>
        )}

        {!loading && !error && debates.length > 0 && (
          <div className="space-y-3">
            {debates.map((d) => {
              const topicLabel = TOPIC_LABELS[d.topic_id];
              return (
                <button
                  key={d.debate_id}
                  onClick={() => navigate(`/debate/${d.debate_id}`)}
                  className="card w-full p-4 text-left hover:ring-1 hover:ring-gray-300 dark:hover:ring-gray-500 transition-all"
                >
                  <div className="flex items-start justify-between gap-4">
                    {/* Left side: topic + condition info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {topicLabel?.title ?? d.topic_id}
                        </span>
                        <StatusBadge status={d.status} />
                      </div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {d.condition_label}
                      </p>
                      <div className="flex items-center gap-4 mt-2 text-xs text-gray-400">
                        <span>
                          Round {d.current_round}/{d.total_rounds}
                        </span>
                        <span>{d.turn_count} turns</span>
                        <span>
                          {d.intervention_count} intervention
                          {d.intervention_count !== 1 ? "s" : ""}
                        </span>
                      </div>
                    </div>

                    {/* Right side: timestamp */}
                    <div className="text-right shrink-0">
                      <span className="text-xs text-gray-400">
                        {formatDate(d.created_at)}
                      </span>
                    </div>
                  </div>

                  {/* Error message if any */}
                  {d.error_message && (
                    <p className="mt-2 text-xs text-red-500 truncate">
                      {d.error_message}
                    </p>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
