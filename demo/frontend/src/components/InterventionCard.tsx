import { useState } from "react";
import type { Intervention } from "../types";
import { ChevronDown, ChevronUp, Shield } from "lucide-react";

interface Props {
  intervention: Intervention;
}

export default function InterventionCard({ intervention }: Props) {
  const [expanded, setExpanded] = useState(false);

  const strategyLabel = intervention.strategy
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  const isHuman = intervention.source === "human" || intervention.strategy === "human_injection";

  return (
    <div className="rounded-lg border-2 border-amber-300 dark:border-amber-600 bg-amber-50 dark:bg-amber-900/20 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Shield size={16} className="text-amber-600" />
          <span className="text-xs font-bold uppercase tracking-wide text-amber-800 dark:text-amber-300">
            Moderator &mdash; {strategyLabel}
          </span>
        </div>
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            isHuman
              ? "bg-purple-100 text-purple-700"
              : "bg-amber-200 text-amber-800"
          }`}
        >
          {isHuman ? "HUMAN" : "AUTOMATED"}
        </span>
      </div>

      {/* Intervention text */}
      <p className="text-sm text-amber-900 dark:text-amber-100 italic leading-relaxed">
        &ldquo;{intervention.intervention_text}&rdquo;
      </p>

      {/* Expandable rationale */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 mt-3 text-xs font-medium text-amber-700 dark:text-amber-400 hover:text-amber-900 transition-colors"
      >
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        SHOW RATIONALE
      </button>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-amber-200 dark:border-amber-700 space-y-2 text-xs text-amber-800 dark:text-amber-200">
          {intervention.trigger_dimension && (
            <div>
              <span className="font-semibold">Trigger:</span>{" "}
              {intervention.trigger_dimension} = {intervention.trigger_score?.toFixed(1)}
            </div>
          )}
          {intervention.moderator_output && (
            <>
              {intervention.moderator_output.diagnosis && (
                <div>
                  <span className="font-semibold">Diagnosis:</span>{" "}
                  {String(intervention.moderator_output.diagnosis)}
                </div>
              )}
              {intervention.moderator_output.key_claims && (
                <div>
                  <span className="font-semibold">Key Claims:</span>{" "}
                  {Array.isArray(intervention.moderator_output.key_claims)
                    ? (intervention.moderator_output.key_claims as string[]).join("; ")
                    : String(intervention.moderator_output.key_claims)}
                </div>
              )}
              {intervention.moderator_output.reframing && (
                <div>
                  <span className="font-semibold">Reframing:</span>{" "}
                  {String(intervention.moderator_output.reframing)}
                </div>
              )}
              <details className="mt-2">
                <summary className="cursor-pointer text-amber-600 hover:text-amber-800">
                  Raw JSON
                </summary>
                <pre className="mt-1 p-2 bg-amber-100 dark:bg-amber-900/40 rounded text-xs overflow-x-auto">
                  {JSON.stringify(intervention.moderator_output, null, 2)}
                </pre>
              </details>
            </>
          )}
        </div>
      )}
    </div>
  );
}
