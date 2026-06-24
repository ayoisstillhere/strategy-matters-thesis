import type {
  ConditionConfig,
  DebateInfo,
  TopicConfig,
  Transcript,
} from "./types";

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// Config
export const getTopics = () => fetchJSON<TopicConfig[]>("/config/topics");
export const getConditions = () =>
  fetchJSON<ConditionConfig[]>("/config/conditions");

// Debates
export const listDebates = () => fetchJSON<DebateInfo[]>("/debates");

export const startDebate = (params: {
  topic_id: string;
  condition_id: string;
  num_rounds: number;
  language: string;
}) =>
  fetchJSON<DebateInfo>("/debate/start", {
    method: "POST",
    body: JSON.stringify(params),
  });

export const getDebateStatus = (id: string) =>
  fetchJSON<DebateInfo>(`/debate/${id}/status`);

export const getTranscript = (id: string) =>
  fetchJSON<Transcript>(`/debate/${id}/transcript`);

export const injectIntervention = (id: string, text: string) =>
  fetchJSON<{ status: string }>(`/debate/${id}/inject-intervention`, {
    method: "POST",
    body: JSON.stringify({ text, source: "human" }),
  });
