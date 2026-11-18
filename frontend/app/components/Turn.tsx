"use client";

import { useState } from "react";
import type { Turn as TurnData } from "@/app/hooks/useAgentStream";
import { Trace } from "./Trace";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const TOOL_LABELS: Record<string, string> = {
  web_search: "web search",
  search_documents: "documents",
  calculator: "calculator",
};

function toolsUsed(turn: TurnData): string[] {
  const names = turn.steps.flatMap((s) => (s.kind === "tool" ? [s.tool] : []));
  return [...new Set(names)].map((n) => TOOL_LABELS[n] ?? n);
}

export function Turn({ turn, latest }: { turn: TurnData; latest: boolean }) {
  const [showReasoning, setShowReasoning] = useState(true);
  const used = toolsUsed(turn);
  const hasReasoning = turn.plan !== null || turn.steps.length > 0;

  // tuck away a turn's reasoning once it is no longer the active one, so the
  // conversation stays readable but the latest run is always open. done while
  // rendering, when latest changes, rather than in an effect that would render twice
  const [wasLatest, setWasLatest] = useState(latest);
  if (latest !== wasLatest) {
    setWasLatest(latest);
    if (!latest) setShowReasoning(false);
  }

  return (
    <div className="rise border-t border-line py-8 first:border-t-0 first:pt-2">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-faint">
          You
        </span>
        <p className="font-serif text-lg leading-snug text-ink">
          {turn.question}
        </p>
      </div>

      {hasReasoning && (
        <div className="mt-5">
          <button
            onClick={() => setShowReasoning((v) => !v)}
            className="font-mono text-[11px] uppercase tracking-[0.14em] text-faint transition-colors hover:text-muted"
          >
            {showReasoning ? "Hide reasoning" : "Show reasoning"}
          </button>
          {showReasoning && <Trace plan={turn.plan} steps={turn.steps} />}
        </div>
      )}

      {turn.error && <p className="mt-4 text-sm text-warn">{turn.error}</p>}

      {turn.answer && (
        <div className="mt-6 overflow-hidden rounded-2xl border border-line bg-surface">
          <div className="border-b border-line px-5 py-2.5">
            <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-accent">
              Answer
            </span>
          </div>
          <div className="px-5 py-4">
            <div className="markdown leading-relaxed text-ink/90">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: (props) => (
                    <a
                      {...props}
                      target="_blank"
                      rel="noreferrer"
                      className="text-accent underline underline-offset-2 hover:opacity-80"
                    />
                  ),
                }}
              >
                {turn.answer}
              </ReactMarkdown>
            </div>
            {used.length > 0 && (
              <p className="mt-4 font-mono text-[11px] text-faint">
                used {used.join(" · ")}
              </p>
            )}
          </div>
        </div>
      )}

      {turn.status === "thinking" && !turn.answer && latest && (
        <div className="mt-5 flex items-center gap-2 text-sm text-faint">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />{" "}
          cooking
        </div>
      )}
    </div>
  );
}
