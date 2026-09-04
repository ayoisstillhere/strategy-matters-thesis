// ---------------------------------------------------------------------------
// Party colors and metadata
// ---------------------------------------------------------------------------

export const PARTY_COLORS: Record<string, string> = {
  "CDU/CSU": "#1A1A1A",
  SPD: "#E3000F",
  "Bündnis 90/Die Grünen": "#64A12D",
  FDP: "#FFED00",
  "Die Linke": "#BE3075",
  AfD: "#009EE0",
};

export const PARTY_SHORT_NAMES: Record<string, string> = {
  "CDU/CSU": "Union (CDU/CSU)",
  SPD: "SPD",
  "Bündnis 90/Die Grünen": "Grüne",
  FDP: "FDP",
  "Die Linke": "Die Linke",
  AfD: "AfD",
};

export const PARTY_IDEOLOGY: Record<string, string> = {
  "CDU/CSU": "Centre-right, market economy",
  SPD: "Centre-left, social democracy",
  "Bündnis 90/Die Grünen": "Green, progressive",
  FDP: "Liberal, free market",
  "Die Linke": "Democratic socialist",
  AfD: "Right-wing populist",
};

// ---------------------------------------------------------------------------
// Score dimensions
// ---------------------------------------------------------------------------

export const DIMENSIONS = [
  { key: "civility", abbrev: "CIV", label: "Civility" },
  { key: "relevance", abbrev: "REL", label: "Relevance" },
  { key: "logical_consistency", abbrev: "LOG", label: "Logical Consistency" },
  { key: "argument_strength", abbrev: "ARG", label: "Argument Strength" },
  { key: "document_grounding", abbrev: "GRD", label: "Document-grounding" },
  { key: "responsiveness", abbrev: "RSP", label: "Responsiveness" },
  { key: "stance_differentiation", abbrev: "STD", label: "Stance Differentiation" },
] as const;

export function getScoreColor(score: number): string {
  if (score >= 4) return "text-green-600 bg-green-50 border-green-200";
  if (score >= 3) return "text-amber-600 bg-amber-50 border-amber-200";
  return "text-red-600 bg-red-50 border-red-200";
}

export function getScoreBarColor(score: number): string {
  if (score >= 4) return "bg-green-500";
  if (score >= 3) return "bg-amber-400";
  return "bg-red-500";
}

// ---------------------------------------------------------------------------
// Topic display labels
// ---------------------------------------------------------------------------

export const TOPIC_LABELS: Record<string, { title: string; subtitle: string }> = {
  mindestlohn: { title: "Mindestlohn", subtitle: "Minimum Wage" },
  rentenpolitik: { title: "Rentenpolitik", subtitle: "Pension Policy" },
  migrationspolitik: { title: "Migrationspolitik", subtitle: "Migration Policy" },
  sozialpolitik: { title: "Sozialpolitik", subtitle: "Wealth Redistribution" },
};

// ---------------------------------------------------------------------------
// API base URL
// ---------------------------------------------------------------------------

export const API_BASE = "/api";
export const WS_BASE =
  window.location.protocol === "https:" ? "wss://" : "ws://" +
  window.location.host;
