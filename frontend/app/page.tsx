"use client";

import { useEffect, useRef, useState } from "react";
import { wakeAgent } from "@/lib/agentStream";
import { Turn } from "./components/Turn";
import { useAgentStream } from "./hooks/useAgentStream";

const EXAMPLES = [
  "What's the population gap between the two most populated Southeast Asian countries, as a percentage of the larger?",
  "Summarise the main points from the NEA annual report.",
  "How many times taller is the Burj Khalifa than the Eiffel Tower?",
];

export default function Home() {
  const { turns, busy, ask, reset } = useAgentStream();
  const [text, setText] = useState("");
  const taRef = useRef<HTMLTextAreaElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // start waking the backend now, so it is up by the time the first question lands
  useEffect(() => {
    wakeAgent();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const grow = () => {
    const el = taRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }
  };

  const send = (q: string) => {
    if (busy || !q.trim()) return;
    ask(q);
    setText("");
    if (taRef.current) taRef.current.style.height = "auto";
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(text);
    }
  };

  const empty = turns.length === 0;

  return (
    <div className="flex min-h-screen flex-col">
      <main className="mx-auto w-full max-w-2xl flex-1 px-5">
        <div className="flex items-start justify-between gap-4 pt-12">
          <h1 className="font-serif text-4xl leading-none text-ink">
            Let It Cook
          </h1>

          <a
            href="https://riskilinardi.vercel.app"
            target="_blank"
            rel="noreferrer"
            className="mt-1 shrink-0 font-mono text-[11px] uppercase tracking-[0.14em] text-faint transition-colors hover:text-muted"
          >
            by RL ↗
          </a>
        </div>
        <p className="mt-4 max-w-lg leading-relaxed text-muted">
          An agent that plans, picks its tools, and reviews its own answer, with
          every step shown. Ask it something and &quot;Let It Cook&quot;.
        </p>

        {empty ? (
          <section className="py-16 text-center">
            <p className="font-serif text-2xl italic leading-snug text-ink/90">
              Ask and watch it reason its way there.
            </p>
            <p className="mx-auto mt-4 max-w-md leading-relaxed text-muted">
              It maps out an approach, chooses between web search, a built-in
              knowledge base, and a calculator, then checks its own draft before
              answering.
            </p>
            <div className="mt-9">
              <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-faint">
                Try one
              </div>
              <div className="mt-3 flex flex-col gap-2">
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    className="rounded-xl border border-line bg-surface px-4 py-3 text-left text-sm text-muted transition-colors hover:border-accent/40 hover:text-ink"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </section>
        ) : (
          <section className="pt-8 pb-10">
            {turns.map((t, i) => (
              <Turn key={t.id} turn={t} latest={i === turns.length - 1} />
            ))}
            <div ref={bottomRef} />
          </section>
        )}
      </main>

      <footer className="sticky bottom-0 border-t border-line bg-canvas/80 backdrop-blur">
        <div className="mx-auto max-w-2xl px-5 py-4">
          <div className="flex items-center gap-2 rounded-2xl border border-line bg-surface px-4 py-3 transition-colors focus-within:border-accent/50">
            <textarea
              ref={taRef}
              value={text}
              rows={1}
              onChange={(e) => {
                setText(e.target.value);
                grow();
              }}
              onKeyDown={onKeyDown}
              placeholder="Ask anything..."
              className="max-h-40 flex-1 resize-none bg-transparent leading-relaxed text-ink placeholder:text-faint focus:outline-none"
            />
            <button
              onClick={() => send(text)}
              disabled={busy || !text.trim()}
              className="shrink-0 rounded-xl bg-accent px-5 py-2 text-sm font-medium text-canvas transition-opacity hover:opacity-90 disabled:opacity-30"
            >
              {busy ? "Cooking" : "Cook"}
            </button>
          </div>
          <div className="mt-2 flex items-center justify-between">
            <span className="font-mono text-[11px] text-faint">
              Enter to send · Shift + Enter for a new line
            </span>
            {!empty && (
              <button
                onClick={reset}
                disabled={busy}
                className="flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] text-muted transition-colors hover:border-accent/40 hover:text-ink disabled:opacity-40"
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="h-3 w-3"
                >
                  <path
                    d="M3 6h18M8 6V4h8v2m-9 0v14a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                Clear chat
              </button>
            )}
          </div>
        </div>
      </footer>
    </div>
  );
}
