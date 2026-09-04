import type { DebateInfo, RoundSummary } from "../types";
import { DIMENSIONS, getScoreBarColor } from "../constants";
import { Download } from "lucide-react";
import { getTranscript } from "../api";
import ScoreChart from "./ScoreChart";

interface Props {
  info: DebateInfo | null;
  roundSummaries: RoundSummary[];
}

export default function ScorePanel({ info, roundSummaries }: Props) {
  const latestSummary = roundSummaries[roundSummaries.length - 1] ?? null;
  const maxInterventions = 3;

  // Progress bar width
  const progress = info
    ? Math.round((info.current_round / info.total_rounds) * 100)
    : 0;

  return (
    <div className="space-y-5">
      {/* Progress */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Current Progress
          </span>
          <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
            Round {info?.current_round ?? 0} of {info?.total_rounds ?? 10}
          </span>
        </div>
        <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-600 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Interventions counter */}
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-600 dark:text-gray-400">Interventions</span>
        <span className="font-semibold text-gray-900 dark:text-white">
          {info?.intervention_count ?? 0}{" "}
          <span className="text-xs text-gray-400">/ {maxInterventions} MAX</span>
        </span>
      </div>

      {/* Discourse Dimensions */}
      <div>
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Discourse Dimensions
        </h3>
        <div className="space-y-2.5">
          {DIMENSIONS.map((dim) => {
            const score = latestSummary?.scores?.[dim.key] ?? 0;
            const widthPct = (score / 5) * 100;
            const isLow = score > 0 && score < 2.5;
            return (
              <div key={dim.key}>
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    {dim.label}
                  </span>
                  <span
                    className={`text-xs font-semibold ${
                      isLow ? "text-red-600" : "text-gray-700 dark:text-gray-300"
                    }`}
                  >
                    {score > 0 ? score.toFixed(1) : "—"}
                  </span>
                </div>
                <div className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${getScoreBarColor(score)}`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Score trajectory chart */}
      {roundSummaries.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Score Trajectory
          </h3>
          <ScoreChart roundSummaries={roundSummaries} height={180} compact />
        </div>
      )}

      {/* Export button */}
      <button
        className="w-full btn-outline flex items-center justify-center gap-2 text-sm mt-4"
        disabled={!info?.debate_id}
        onClick={async () => {
          if (!info?.debate_id) return;
          try {
            const transcript = await getTranscript(info.debate_id);
            const blob = new Blob([JSON.stringify(transcript, null, 2)], {
              type: "application/json",
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `debate-${info.debate_id.slice(0, 8)}.json`;
            a.click();
            URL.revokeObjectURL(url);
          } catch (e) {
            console.error("Export failed:", e);
          }
        }}
      >
        <Download size={14} />
        Export Session Data
      </button>
    </div>
  );
}
