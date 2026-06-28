"use client";

import { useCallback, useReducer, useRef } from "react";
import { AgentEvent, streamAgent } from "@/lib/agentStream";

export type ToolStep = {
  kind: "tool";
  tool: string;
  input: Record<string, unknown>;
  result: string | null;
};
export type ReflectionStep = {
  kind: "reflection";
  note: string;
  approved: boolean;
};
export type Step = ToolStep | ReflectionStep;

export type Turn = {
  id: string;
  question: string;
  plan: string | null;
  steps: Step[];
  answer: string | null;
  status: "thinking" | "done" | "error";
  error: string | null;
};

type Action =
  | { kind: "ask"; id: string; question: string }
  | { kind: "event"; event: AgentEvent }
  | { kind: "error"; message: string }
  | { kind: "clear" };

function patchLast(turns: Turn[], patch: (t: Turn) => Turn): Turn[] {
  if (turns.length === 0) return turns;
  const copy = turns.slice();
  copy[copy.length - 1] = patch(copy[copy.length - 1]);
  return copy;
}

function reducer(turns: Turn[], action: Action): Turn[] {
  switch (action.kind) {
    case "clear":
      return [];
    case "ask":
      return [
        ...turns,
        {
          id: action.id,
          question: action.question,
          plan: null,
          steps: [],
          answer: null,
          status: "thinking",
          error: null,
        },
      ];
    case "error":
      return patchLast(turns, (t) => ({
        ...t,
        status: "error",
        error: action.message,
      }));
    case "event": {
      const e = action.event;
      return patchLast(turns, (t) => {
        switch (e.type) {
          case "plan":
            return { ...t, plan: e.plan };
          case "tool_call":
            return {
              ...t,
              steps: [
                ...t.steps,
                { kind: "tool", tool: e.tool, input: e.input, result: null },
              ],
            };
          case "observation": {
            const steps = t.steps.slice();
            const i = steps.findIndex(
              (s) =>
                s.kind === "tool" && s.tool === e.tool && s.result === null,
            );
            if (i !== -1)
              steps[i] = { ...(steps[i] as ToolStep), result: e.result };
            return { ...t, steps };
          }
          case "reflection":
            return {
              ...t,
              steps: [
                ...t.steps,
                { kind: "reflection", note: e.note, approved: e.approved },
              ],
            };
          case "answer":
            return { ...t, answer: e.answer };
          case "done":
            return { ...t, status: "done" };
          default:
            return t;
        }
      });
    }
    default:
      return turns;
  }
}

export function useAgentStream() {
  const [turns, dispatch] = useReducer(reducer, []);
  const threadId = useRef<string>("");
  const controller = useRef<AbortController | null>(null);
  if (!threadId.current) threadId.current = crypto.randomUUID();

  const busy =
    turns.length > 0 && turns[turns.length - 1].status === "thinking";

  const ask = useCallback(async (raw: string) => {
    const question = raw.trim();
    if (!question) return;

    controller.current?.abort();
    controller.current = new AbortController();
    dispatch({ kind: "ask", id: crypto.randomUUID(), question });

    try {
      for await (const event of streamAgent(
        question,
        threadId.current,
        controller.current.signal,
      )) {
        dispatch({ kind: "event", event });
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        dispatch({ kind: "error", message: (err as Error).message });
      }
    }
  }, []);

  const reset = useCallback(() => {
    controller.current?.abort();
    // Generate a new thread ID for the next conversation.
    threadId.current = crypto.randomUUID();
    dispatch({ kind: "clear" });
  }, []);

  return { turns, busy, ask, reset };
}
