// ---------------------------------------------------------------------------
// API types matching backend schemas
// ---------------------------------------------------------------------------

export type DebateStatus = "pending" | "running" | "completed" | "error";

export interface DebateInfo {
  debate_id: string;
  topic_id: string;
  condition_id: string;
  condition_label: string;
  status: DebateStatus;
  current_round: number;
  total_rounds: number;
  turn_count: number;
  intervention_count: number;
  error_message: string | null;
}

export interface Turn {
  turn_id: string;
  round_number: number;
  turn_in_round: number;
  agent_name: string;
  text: string;
  scores: Record<string, number> | null;
  timestamp: string | null;
}

export interface Intervention {
  intervention_id: string;
  round_number: number;
  source: string;
  strategy: string;
  trigger_dimension: string | null;
  trigger_score: number | null;
  silent_control: boolean;
  intervention_text: string;
  moderator_output: Record<string, unknown> | null;
  timestamp: string | null;
}

export interface Transcript {
  debate_id: string;
  status: DebateStatus;
  config: Record<string, unknown> | null;
  turns: Turn[];
  interventions: Intervention[];
  round_summaries: RoundSummary[];
}

export interface RoundSummary {
  round_number: number;
  composite: number;
  plateau: boolean;
  scores: Record<string, number>;
}

export interface TopicConfig {
  id: string;
  title: string;
  type: string;
  framing_prompt: string;
}

export interface ConditionConfig {
  id: string;
  label: string;
  type: string;
  has_moderator: boolean;
  uses_trigger: boolean;
  trigger_strategy: string | null;
}

// WebSocket events
export type WSEventType =
  | "turn"
  | "intervention"
  | "round_summary"
  | "status_change"
  | "error";

export interface WSEvent {
  event_type: WSEventType;
  data: Record<string, unknown>;
}

// Timeline item (for rendering transcript in order)
export type TimelineItem =
  | { type: "turn"; data: Turn }
  | { type: "intervention"; data: Intervention }
  | { type: "round_start"; roundNumber: number };
