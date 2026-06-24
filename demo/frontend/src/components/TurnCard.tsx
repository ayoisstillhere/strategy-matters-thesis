import { useState } from "react";
import type { Turn } from "../types";
import { PARTY_COLORS, PARTY_SHORT_NAMES, DIMENSIONS, getScoreColor } from "../constants";

interface Props {
  turn: Turn;
}

export default function TurnCard({ turn }: Props) {
  const [showScores, setShowScores] = useState(false);
  const color = PARTY_COLORS[turn.agent_name] || "#6B7280";
  const displayName = PARTY_SHORT_NAMES[turn.agent_name] || turn.agent_name;

  const timeStr = turn.timestamp
    ? new Date(turn.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

  return (
    <div
      className="card p-4 border-l-4 hover:shadow-md transition-shadow cursor-pointer"
      style={{ borderLeftColor: color }}
      onClick={() => setShowScores(!showScores)}
    >
      {/* Header: party name + timestamp */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold uppercase tracking-wide" style={{ color }}>
          {displayName}
        </span>
        <span className="text-xs text-gray-400">{timeStr}</span>
      </div>

      {/* Turn text */}
      <p className="text-sm text-gray-800 dark:text-gray-200 leading-relaxed">
        {turn.text}
      </p>

      {/* Score badges (shown on click/hover) */}
      {showScores && turn.scores && (
        <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
          {DIMENSIONS.map((dim) => {
            const score = turn.scores![dim.key];
            if (score === undefined) return null;
            return (
              <span
                key={dim.key}
                className={`text-xs px-2 py-0.5 rounded-full border font-medium ${getScoreColor(score)}`}
                title={dim.label}
              >
                {dim.abbrev} {score}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
