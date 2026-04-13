import { useCallback, useEffect, useRef, useState } from "react";
import { MessageSquare, Plus, Send, Square, Loader2, Wrench } from "lucide-react";
import { api } from "@/lib/api";
import type { SessionInfo, SessionMessage } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { Markdown } from "@/components/Markdown";
import { Button } from "@/components/ui/button";

interface ToolCall {
  name: string;
  args?: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCall[];
  toolProgress?: Array<{ tool: string; emoji: string; label: string }>;
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Guard against stale session loads racing with newer clicks
  const loadGenRef = useRef(0);

  const refreshSessions = useCallback(() => {
    api.getSessionsWithChildren(30, 0).then((data) => setSessions(data.sessions)).catch(() => {});
  }, []);

  useEffect(() => { refreshSessions(); }, [refreshSessions]);

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const loadSession = useCallback(async (sessionId: string) => {
    // P1: Cancel any active stream before switching chats
    cancelStream();

    // P2: Generation counter guards against out-of-order responses
    const gen = ++loadGenRef.current;
    setCurrentSessionId(sessionId);
    setIsStreaming(false);

    try {
      const data = await api.getSessionMessages(sessionId);
      if (loadGenRef.current !== gen) return; // stale — a newer click superseded us

      // P3: Preserve tool_calls on assistant messages so reopened chats
      // show which tools were invoked (not just empty bubbles)
      const chatMsgs: ChatMessage[] = data.messages
        .filter((m: SessionMessage) => m.role === "user" || m.role === "assistant")
        .map((m: SessionMessage) => {
          const msg: ChatMessage = {
            role: m.role as "user" | "assistant",
            content: typeof m.content === "string" ? m.content : "",
          };
          if (m.tool_calls && m.tool_calls.length > 0) {
            msg.toolCalls = m.tool_calls.map((tc) => ({
              name: tc.function.name,
              args: tc.function.arguments,
            }));
          }
          return msg;
        });
      setMessages(chatMsgs);
    } catch {
      if (loadGenRef.current === gen) setMessages([]);
    }
  }, [cancelStream]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: isStreaming ? "auto" : "smooth" });
  }, [messages, isStreaming]);

  useEffect(() => {
    if (!isStreaming) inputRef.current?.focus();
  }, [isStreaming]);

  const startNewChat = () => {
    cancelStream();
    setCurrentSessionId(null);
    setMessages([]);
    setIsStreaming(false);
    inputRef.current?.focus();
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    const assistantMsg: ChatMessage = { role: "assistant", content: "", toolProgress: [] };
    setMessages((prev) => [...prev, assistantMsg]);

    const controller = new AbortController();
    abortRef.current = controller;

    // Capture the session ID at send time so delta handlers can detect
    // if the user switched chats while this request was in flight.
    const sendSessionId = currentSessionId;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sendSessionId }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const errText = await res.text().catch(() => "Unknown error");
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: `Error: ${errText}` };
          return updated;
        });
        setIsStreaming(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              if (currentEvent === "delta" || (!currentEvent && parsed.text)) {
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  updated[updated.length - 1] = { ...last, content: last.content + (parsed.text || "") };
                  return updated;
                });
              } else if (currentEvent === "hermes.tool.progress") {
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  const tools = [...(last.toolProgress || []), parsed];
                  updated[updated.length - 1] = { ...last, toolProgress: tools };
                  return updated;
                });
              } else if (currentEvent === "done") {
                // Session ID from agent (may have rotated during compression)
                if (parsed.session_id) {
                  setCurrentSessionId(parsed.session_id);
                }
              }
            } catch {
              // Ignore malformed JSON
            }
            currentEvent = "";
          } else if (line.trim() === "") {
            currentEvent = "";
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && !last.content) {
            updated[updated.length - 1] = { role: "assistant", content: `Error: ${(err as Error).message}` };
          }
          return updated;
        });
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
      refreshSessions();
    }
  };

  const stopStreaming = () => {
    cancelStream();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex overflow-hidden -mx-6 -my-8" style={{ height: "calc(100vh - 3rem)" }}>
      {/* Sidebar */}
      <div className="w-48 shrink-0 border-r border-border flex flex-col bg-background">
        <div className="px-2 py-1.5 border-b border-border">
          <Button onClick={startNewChat} variant="outline" className="w-full gap-1.5 text-[0.7rem] h-7">
            <Plus className="h-3 w-3" /> New Chat
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => loadSession(s.id)}
              className={`w-full text-left px-2 py-1.5 border-b border-border/30 transition-colors ${
                currentSessionId === s.id
                  ? "bg-primary/10 text-foreground"
                  : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
              }`}
            >
              <div className="text-[0.7rem] font-medium truncate leading-tight">
                {s.title || s.preview || "Untitled"}
              </div>
              <div className="text-[0.6rem] text-muted-foreground leading-tight">
                {timeAgo(s.last_active || s.started_at)}
              </div>
            </button>
          ))}
          {sessions.length === 0 && (
            <div className="p-3 text-center text-[0.7rem] text-muted-foreground">
              No sessions yet
            </div>
          )}
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {messages.length === 0 && (
            <div className="flex-1 flex items-center justify-center h-full">
              <div className="text-center text-muted-foreground">
                <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-30" />
                <p className="text-xs">Start a conversation with Hermes</p>
              </div>
            </div>
          )}
          <div className="max-w-3xl mx-auto space-y-3">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[75%] rounded-lg px-3 py-2 ${
                  msg.role === "user"
                    ? "bg-primary/10 text-foreground"
                    : "bg-secondary/40 text-foreground"
                }`}
              >
                {msg.role === "assistant" ? (
                  <>
                    {/* Live tool progress from SSE events */}
                    {msg.toolProgress && msg.toolProgress.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mb-2">
                        {msg.toolProgress.map((tp, j) => (
                          <span
                            key={j}
                            className="inline-flex items-center gap-1 rounded-md bg-warning/10 px-2 py-0.5 text-[0.65rem] text-warning border border-warning/20"
                          >
                            {tp.emoji} {tp.tool}
                          </span>
                        ))}
                      </div>
                    )}
                    {/* Persisted tool_calls from session history */}
                    {msg.toolCalls && msg.toolCalls.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mb-2">
                        {msg.toolCalls.map((tc, j) => (
                          <span
                            key={j}
                            className="inline-flex items-center gap-1 rounded-md bg-secondary/60 px-2 py-0.5 text-[0.65rem] text-muted-foreground border border-border"
                          >
                            <Wrench className="h-2.5 w-2.5" /> {tc.name}
                          </span>
                        ))}
                      </div>
                    )}
                    {msg.content ? (
                      <Markdown content={msg.content} />
                    ) : msg.toolCalls && msg.toolCalls.length > 0 ? (
                      // Tool-call-only turn — don't show empty bubble, tools are above
                      null
                    ) : isStreaming && i === messages.length - 1 ? (
                      <div className="flex items-center gap-2 text-muted-foreground text-xs">
                        <Loader2 className="h-3 w-3 animate-spin" /> Thinking...
                      </div>
                    ) : null}
                  </>
                ) : (
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>
            </div>
          ))}
          </div>
          <div ref={messagesEndRef} />
        </div>

        <div className="border-t border-border px-3 py-2">
          <div className="flex gap-2 items-end max-w-3xl mx-auto">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Hermes..."
              disabled={isStreaming}
              rows={1}
              className="flex-1 resize-none rounded-lg border border-border bg-secondary/30 px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
              style={{ minHeight: "2.5rem", maxHeight: "10rem" }}
              onInput={(e) => {
                const el = e.target as HTMLTextAreaElement;
                el.style.height = "auto";
                el.style.height = Math.min(el.scrollHeight, 160) + "px";
              }}
            />
            {isStreaming ? (
              <Button onClick={stopStreaming} variant="outline" size="sm" className="shrink-0 h-10 w-10 p-0">
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button onClick={sendMessage} disabled={!input.trim()} size="sm" className="shrink-0 h-10 w-10 p-0">
                <Send className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
