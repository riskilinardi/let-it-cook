export type AgentEvent =
  | { type: "plan"; plan: string }
  | { type: "tool_call"; tool: string; input: Record<string, unknown> }
  | { type: "observation"; tool: string; result: string }
  | { type: "reflection"; note: string; approved: boolean }
  | { type: "answer"; answer: string }
  | { type: "done" };

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function messageForStatus(status: number): string {
  if (status === 429)
    return "The demo is busy right now. Give it a few seconds and try again.";
  if (status === 503)
    return "The agent is warming up or briefly overloaded. Try again in a moment.";
  if (status >= 500)
    return "The server had a problem. Please try again shortly.";
  return "Something went wrong reaching the agent. Please try again.";
}

export async function* streamAgent(
  question: string,
  threadId: string,
  signal?: AbortSignal,
): AsyncGenerator<AgentEvent> {
  const response = await fetch(`${API_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(messageForStatus(response.status));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Split the buffer into parts separated by double newlines, which indicates the end of an event.
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data: "));
      if (line) {
        yield JSON.parse(line.slice("data: ".length)) as AgentEvent;
      }
    }
  }
}
