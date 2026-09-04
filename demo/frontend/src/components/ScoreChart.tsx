import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { RoundSummary } from "../types";
import { DIMENSIONS } from "../constants";

interface Props {
  roundSummaries: RoundSummary[];
  height?: number;
  compact?: boolean;
}

const DIMENSION_COLORS: Record<string, string> = {
  civility: "#22c55e",
  relevance: "#3b82f6",
  logical_consistency: "#a855f7",
  argument_strength: "#f59e0b",
  document_grounding: "#ef4444",
  responsiveness: "#06b6d4",
  stance_differentiation: "#ec4899",
};

export default function ScoreChart({
  roundSummaries,
  height = 220,
  compact = false,
}: Props) {
  if (roundSummaries.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-xs text-gray-400"
        style={{ height }}
      >
        Waiting for round data...
      </div>
    );
  }

  const data = roundSummaries.map((s) => ({
    round: `R${s.round_number}`,
    ...s.scores,
  }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis
            dataKey="round"
            tick={{ fontSize: 10 }}
            interval={0}
          />
          <YAxis
            domain={[1, 5]}
            ticks={[1, 2, 3, 4, 5]}
            tick={{ fontSize: 10 }}
          />
          <Tooltip
            contentStyle={{
              fontSize: 11,
              borderRadius: 8,
              border: "1px solid #e5e7eb",
            }}
            formatter={(value: number, name: string) => {
              const dim = DIMENSIONS.find((d) => d.key === name);
              return [value.toFixed(2), dim?.label ?? name];
            }}
          />
          {DIMENSIONS.map((dim) => (
            <Line
              key={dim.key}
              type="monotone"
              dataKey={dim.key}
              stroke={DIMENSION_COLORS[dim.key] ?? "#6b7280"}
              strokeWidth={1.5}
              dot={{ r: 2 }}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Legend */}
      {!compact && (
        <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1 px-1">
          {DIMENSIONS.map((dim) => (
            <span key={dim.key} className="flex items-center gap-1 text-[10px] text-gray-600">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: DIMENSION_COLORS[dim.key] }}
              />
              {dim.abbrev}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
