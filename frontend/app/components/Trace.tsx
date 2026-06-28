"use client";

import { useState } from "react";
import type { Step, ToolStep } from "@/app/hooks/useAgentStream";

const TOOLS: Record<string, { label: string; dot: string; ink: string }> = {
  web_search: { label: "Web search", dot: "bg-tool-web", ink: "text-tool-web" },
  search_documents: {
    label: "Documents",
    dot: "bg-tool-docs",
    ink: "text-tool-docs",
  },
  calculator: {
    label: "Calculator",
    dot: "bg-tool-calc",
    ink: "text-tool-calc",
  },
};

function toolInfo(name: string) {
  return TOOLS[name] ?? { label: name, dot: "bg-muted", ink: "text-muted" };
}

function summarise(input: Record<string, unknown>): string {
  if (typeof input.query === "string") return input.query;
  if (typeof input.expression === "string") return input.expression;
  return JSON.stringify(input);
}

function Label({
  className = "",
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`font-mono text-[11px] uppercase tracking-[0.14em] ${className}`}
    >
      {children}
    </div>
  );
}

function Node({ dot, children }: { dot: string; children: React.ReactNode }) {
  return (
    <div className="rise relative pb-6 pl-7">
      <span
        className={`absolute left-0 top-[7px] h-2 w-2 rounded-full ring-4 ring-canvas ${dot}`}
      />
      {children}
    </div>
  );
}

function ToolNode({ step }: { step: ToolStep }) {
  const [open, setOpen] = useState(false);
  const t = toolInfo(step.tool);
  return (
    <Node dot={t.dot}>
      <Label className={t.ink}>{t.label}</Label>
      <div className="mt-1 font-mono text-[13px] leading-relaxed text-muted">
        {summarise(step.input)}
      </div>
      {step.result === null ? (
        <div className="mt-1.5 flex items-center gap-2 text-[13px] text-faint">
          <span className="h-1 w-1 animate-pulse rounded-full bg-faint" />{" "}
          working
        </div>
      ) : (
        <div className="mt-1.5">
          <button
            onClick={() => setOpen((v) => !v)}
            className="text-[13px] text-faint transition-colors hover:text-muted"
          >
            {open ? "Hide result" : "View result"}
          </button>
          {open && (
            <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-canvas/60 p-3 font-mono text-xs leading-relaxed text-muted">
              {step.result}
            </pre>
          )}
        </div>
      )}
    </Node>
  );
}

function ReflectionNode({
  approved,
  note,
}: {
  approved: boolean;
  note: string;
}) {
  return (
    <Node dot={approved ? "bg-positive" : "bg-warn"}>
      <Label className={approved ? "text-positive" : "text-warn"}>
        Self-check · {approved ? "approved" : "needs work"}
      </Label>
      <div className="mt-1 text-[13px] leading-relaxed text-muted">{note}</div>
    </Node>
  );
}

export function Trace({ plan, steps }: { plan: string | null; steps: Step[] }) {
  if (!plan && steps.length === 0) return null;
  return (
    <div className="relative mt-4">
      <span className="absolute bottom-3 left-[3px] top-2 w-px bg-line" />
      {plan && (
        <Node dot="bg-ink">
          <Label className="text-muted">Plan</Label>
          <div className="mt-1 whitespace-pre-wrap text-[13px] leading-relaxed text-muted">
            {plan}
          </div>
        </Node>
      )}
      {steps.map((s, i) =>
        s.kind === "tool" ? (
          <ToolNode key={i} step={s} />
        ) : (
          <ReflectionNode key={i} approved={s.approved} note={s.note} />
        ),
      )}
    </div>
  );
}
