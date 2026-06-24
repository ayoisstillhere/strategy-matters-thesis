import { useState, useMemo } from "react";
import type { Intervention } from "../types";
import { ChevronDown, ChevronUp, Shield } from "lucide-react";

interface Props {
  intervention: Intervention;
}

/** Try to extract clean text from a potentially raw-JSON intervention_text. */
function cleanInterventionText(text: string): string {
  let s = text.trim();
  // Strip markdown code fences
  if (s.startsWith("```")) {
    const firstNl = s.indexOf("\n");
    if (firstNl !== -1) s = s.slice(firstNl + 1);
    if (s.endsWith("```")) s = s.slice(0, -3);
    s = s.trim();
  }
  // If it looks like JSON, try to extract instruction_for_next_round or intervention_text
  if (s.startsWith("{")) {
    try {
      const obj = JSON.parse(s);
      return (
        obj.instruction_for_next_round ??
        obj.intervention_text ??
        obj.consensus_statement ??
        s
      );
    } catch {
      // Might be truncated JSON — extract any readable instruction field
      const match = s.match(/"instruction_for_next_round"\s*:\s*"([^"]+)/);
      if (match) return match[1] + "…";
      const match2 = s.match(/"intervention_text"\s*:\s*"([^"]+)/);
      if (match2) return match2[1] + "…";
    }
  }
  return s;
}

/** Try to parse a structured object from raw intervention_text when outputs are null. */
function parseFallbackOutput(text: string): Record<string, unknown> | null {
  let s = text.trim();
  if (s.startsWith("```")) {
    const firstNl = s.indexOf("\n");
    if (firstNl !== -1) s = s.slice(firstNl + 1);
    if (s.endsWith("```")) s = s.slice(0, -3);
    s = s.trim();
  }
  if (!s.startsWith("{")) return null;
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

export default function InterventionCard({ intervention }: Props) {
  const [expanded, setExpanded] = useState(false);

  const strategyLabel = intervention.strategy
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  const isHuman = intervention.source === "human" || intervention.strategy === "human_injection";
  const isHabermas = intervention.strategy === "habermas";

  // Resolve the structured output — prefer explicit fields, fall back to parsing raw text
  const output = useMemo(() => {
    if (intervention.habermas_output) return intervention.habermas_output;
    if (intervention.moderator_output) return intervention.moderator_output;
    return parseFallbackOutput(intervention.intervention_text);
  }, [intervention]);

  const displayText = useMemo(
    () => cleanInterventionText(intervention.intervention_text),
    [intervention.intervention_text],
  );

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
        &ldquo;{displayText}&rdquo;
      </p>

      {/* Expandable rationale */}
      {(output || intervention.trigger_dimension) && (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 mt-3 text-xs font-medium text-amber-700 dark:text-amber-400 hover:text-amber-900 transition-colors"
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {expanded ? "HIDE RATIONALE" : "SHOW RATIONALE"}
          </button>

          {expanded && (
            <div className="mt-3 pt-3 border-t border-amber-200 dark:border-amber-700 space-y-2 text-xs text-amber-800 dark:text-amber-200">
              {intervention.trigger_dimension && (
                <div>
                  <span className="font-semibold">Trigger:</span>{" "}
                  {intervention.trigger_dimension} = {intervention.trigger_score?.toFixed(1)}
                </div>
              )}

              {/* Habermas-specific rationale */}
              {isHabermas && output && (
                <>
                  {output.round_summary && (
                    <div>
                      <span className="font-semibold">Round Summary:</span>{" "}
                      {String(output.round_summary)}
                    </div>
                  )}
                  {output.areas_of_agreement && (
                    <div>
                      <span className="font-semibold">Areas of Agreement:</span>{" "}
                      {String(output.areas_of_agreement)}
                    </div>
                  )}
                  {output.areas_of_disagreement && (
                    <div>
                      <span className="font-semibold">Areas of Disagreement:</span>{" "}
                      {String(output.areas_of_disagreement)}
                    </div>
                  )}
                  {output.consensus_statement && (
                    <div>
                      <span className="font-semibold">Consensus Statement:</span>{" "}
                      {String(output.consensus_statement)}
                    </div>
                  )}
                </>
              )}

              {/* Strategy-specific rationale */}
              {!isHabermas && output && (
                <>
                  {output.diagnosis && (
                    <div>
                      <span className="font-semibold">Diagnosis:</span>{" "}
                      {String(output.diagnosis)}
                    </div>
                  )}
                  {output.target_parties && (
                    <div>
                      <span className="font-semibold">Target Parties:</span>{" "}
                      {Array.isArray(output.target_parties)
                        ? (output.target_parties as string[]).join(", ")
                        : String(output.target_parties)}
                    </div>
                  )}
                  {output.target_claim && (
                    <div>
                      <span className="font-semibold">Target Claim:</span>{" "}
                      {String(output.target_claim)}
                    </div>
                  )}
                  {output.expected_next_turn_behaviour && (
                    <div>
                      <span className="font-semibold">Expected Behaviour:</span>{" "}
                      {String(output.expected_next_turn_behaviour)}
                    </div>
                  )}
                </>
              )}

              {/* Raw JSON toggle */}
              {output && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-amber-600 hover:text-amber-800">
                    Raw JSON
                  </summary>
                  <pre className="mt-1 p-2 bg-amber-100 dark:bg-amber-900/40 rounded text-xs overflow-x-auto whitespace-pre-wrap break-words">
                    {JSON.stringify(output, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
