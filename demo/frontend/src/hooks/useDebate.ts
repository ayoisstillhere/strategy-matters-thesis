import { useCallback, useEffect, useRef, useState } from "react";
import type {
  DebateInfo,
  Intervention,
  RoundSummary,
  TimelineItem,
  Turn,
  WSEvent,
} from "../types";
import { getDebateStatus, getTranscript } from "../api";

/**
 * Hook to manage a running debate session:
 * - Connects WebSocket for real-time events
 * - Maintains timeline (turns + interventions in order)
 * - Tracks round summaries and status
 */
export function useDebate(debateId: string | null) {
  const [info, setInfo] = useState<DebateInfo | null>(null);
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [roundSummaries, setRoundSummaries] = useState<RoundSummary[]>([]);
  const [debateConfig, setDebateConfig] = useState<Record<string, unknown> | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const currentRoundRef = useRef(0);

  // Load existing state when debateId changes
  useEffect(() => {
    if (!debateId) return;

    getDebateStatus(debateId).then(setInfo).catch(console.error);
    getTranscript(debateId).then((t) => {
      const items: TimelineItem[] = [];
      let lastRound = 0;
      for (const turn of t.turns) {
        if (turn.round_number > lastRound) {
          lastRound = turn.round_number;
          items.push({ type: "round_start", roundNumber: lastRound });
        }
        items.push({ type: "turn", data: turn });
        // Insert any interventions that belong after this turn
        const intervention = t.interventions.find(
          (i) => i.round_number === turn.round_number && !i.silent_control
        );
        if (intervention && turn.turn_in_round === 6) {
          items.push({ type: "intervention", data: intervention });
        }
      }
      // Also add inline interventions
      for (const i of t.interventions) {
        if (!i.silent_control && !items.some(
          (item) => item.type === "intervention" && item.data.intervention_id === i.intervention_id
        )) {
          items.push({ type: "intervention", data: i });
        }
      }
      setTimeline(items);
      setRoundSummaries(t.round_summaries);
      setDebateConfig(t.config);
      currentRoundRef.current = lastRound;
    }).catch(console.error);
  }, [debateId]);

  // WebSocket connection
  useEffect(() => {
    if (!debateId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/debate/${debateId}/ws`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (event) => {
      const msg: WSEvent = JSON.parse(event.data);

      switch (msg.event_type) {
        case "turn": {
          const turn = msg.data as unknown as Turn;
          setTimeline((prev) => {
            // Add round separator if new round
            const items = [...prev];
            if (turn.round_number > currentRoundRef.current) {
              currentRoundRef.current = turn.round_number;
              items.push({ type: "round_start", roundNumber: turn.round_number });
            }
            items.push({ type: "turn", data: turn });
            return items;
          });
          setInfo((prev) =>
            prev
              ? { ...prev, turn_count: prev.turn_count + 1, current_round: turn.round_number }
              : prev
          );
          break;
        }
        case "intervention": {
          const intervention = msg.data as unknown as Intervention;
          if (!intervention.silent_control) {
            setTimeline((prev) => [...prev, { type: "intervention", data: intervention }]);
            setInfo((prev) =>
              prev ? { ...prev, intervention_count: prev.intervention_count + 1 } : prev
            );
          }
          break;
        }
        case "round_summary": {
          const summary = msg.data as unknown as RoundSummary;
          setRoundSummaries((prev) => [...prev, summary]);
          break;
        }
        case "status_change": {
          const status = msg.data.status as string;
          setInfo((prev) => (prev ? { ...prev, status: status as DebateInfo["status"] } : prev));
          break;
        }
        case "error": {
          setInfo((prev) =>
            prev
              ? { ...prev, status: "error", error_message: msg.data.error as string }
              : prev
          );
          break;
        }
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [debateId]);

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send("ping");
    }
  }, []);

  return { info, timeline, roundSummaries, debateConfig, connected, sendPing };
}
