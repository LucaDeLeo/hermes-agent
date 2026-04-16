import { memo, useCallback, useEffect, useRef, useState } from "react";
import {
  MessageSquare,
  Plus,
  Send,
  Square,
  Loader2,
  ChevronRight,
  ChevronDown,
  ArrowDown,
} from "lucide-react";
import { api } from "@/lib/api";
import type { SessionInfo } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { Markdown } from "@/components/Markdown";
import { ThinkingBlock } from "@/components/ThinkingBlock";
import { CopyButton } from "@/components/CopyButton";
import { Button } from "@/components/ui/button";

// Interleaved message model — tool calls and thinking are separate entries between text segments
type ChatMessage =
  | { type: "user"; content: string }
  | { type: "text"; content: string }
  | { type: "thinking"; content: string; isStreaming?: boolean; durationMs?: number }
  | {
      type: "tool";
      name: string;
      toolCallId?: string;
      emoji?: string;
      input: string;
      output: string;
      isError?: boolean;
      duration?: number;
      isStreaming?: boolean;
    };

const OUTPUT_LINE_THRESHOLD = 40;

const ToolCard = memo(function ToolCard({ msg }: { msg: Extract<ChatMessage, { type: "tool" }> }) {
  const [expanded, setExpanded] = useState(false);
  const [showFullOutput, setShowFullOutput] = useState(false);
  const durationLabel = msg.duration != null ? `${msg.duration.toFixed(1)}s` : "";

  const outputLines = msg.output ? msg.output.split("\n") : [];
  const outputTruncated = outputLines.length > OUTPUT_LINE_THRESHOLD;
  const displayedOutput =
    outputTruncated && !showFullOutput
      ? outputLines.slice(0, OUTPUT_LINE_THRESHOLD).join("\n")
      : msg.output;

  return (
    <div className="my-1.5 max-w-3xl mx-auto">
      <button
        onClick={() => setExpanded(!expanded)}
        className={`relative w-full text-left flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[0.7rem] border transition-colors cursor-pointer ${
          msg.isError
            ? "bg-destructive/10 border-destructive/30 text-destructive"
            : msg.isStreaming
            ? "bg-warning/10 border-warning/20 text-warning"
            : "bg-secondary/50 border-border text-muted-foreground hover:bg-secondary/70"
        }`}
      >
        {expanded ? (
          <ChevronDown className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0" />
        )}
        <span>{msg.emoji || "⚡"}</span>
        <span className="font-medium">{msg.name}</span>
        {durationLabel && <span className="ml-auto opacity-60">{durationLabel}</span>}
        {msg.isStreaming && <Loader2 className="h-3 w-3 ml-auto animate-spin" />}
        {msg.isStreaming && (
          <span className="absolute bottom-0 left-0 right-0 h-[1.5px] overflow-hidden">
            <span className="block h-full w-1/3 bg-warning/70 animate-[slide_1.2s_ease-in-out_infinite]" />
          </span>
        )}
      </button>
      {expanded && (
        <div className="mt-1 rounded-md border border-border bg-secondary/30 text-[0.65rem] overflow-hidden">
          {msg.input ? (
            <div>
              <div className="px-2.5 py-1 bg-secondary/40 text-muted-foreground font-medium border-b border-border">
                Input
              </div>
              <pre className="px-2.5 py-2 overflow-x-auto font-mono leading-relaxed whitespace-pre-wrap break-all max-h-60 overflow-y-auto text-foreground">
                {formatToolData(msg.input)}
              </pre>
            </div>
          ) : msg.isStreaming ? (
            <div className="px-2.5 py-2 text-muted-foreground flex items-center gap-1.5">
              <Loader2 className="h-3 w-3 animate-spin" /> Running...
            </div>
          ) : null}
          {msg.output && (
            <div className={msg.input ? "border-t border-border" : ""}>
              <div className="px-2.5 py-1 bg-secondary/40 text-muted-foreground font-medium border-b border-border flex items-center">
                <span>Output</span>
                <CopyButton value={msg.output} className="ml-auto h-5" />
              </div>
              <pre className="px-2.5 py-2 overflow-x-auto font-mono leading-relaxed whitespace-pre-wrap break-all max-h-80 overflow-y-auto text-foreground">
                {displayedOutput}
              </pre>
              {outputTruncated && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowFullOutput((v) => !v);
                  }}
                  className="w-full px-2.5 py-1 bg-secondary/40 text-muted-foreground hover:text-foreground font-medium border-t border-border text-[0.65rem] cursor-pointer transition-colors"
                >
                  {showFullOutput
                    ? "Collapse"
                    : `Show full (${outputLines.length} lines)`}
                </button>
              )}
            </div>
          )}
          {!msg.input && !msg.output && !msg.isStreaming && (
            <div className="px-2.5 py-2 text-muted-foreground italic">No data captured</div>
          )}
        </div>
      )}
    </div>
  );
});

function PendingDots() {
  return (
    <div className="flex items-center gap-2 text-muted-foreground text-xs">
      <span className="inline-flex gap-1">
        <span className="h-1.5 w-1.5 rounded-full bg-current opacity-50 animate-[bounce_1.1s_ease-in-out_infinite]" />
        <span className="h-1.5 w-1.5 rounded-full bg-current opacity-50 animate-[bounce_1.1s_ease-in-out_0.15s_infinite]" />
        <span className="h-1.5 w-1.5 rounded-full bg-current opacity-50 animate-[bounce_1.1s_ease-in-out_0.3s_infinite]" />
      </span>
      <span className="italic">working</span>
    </div>
  );
}

function formatToolData(data: string): string {
  try {
    const parsed = JSON.parse(data);
    return JSON.stringify(parsed, null, 2);
  } catch {
    // Try parsing Python dict repr: {'key': 'value'} → valid display
    return data;
  }
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const loadGenRef = useRef(0);
  const isAtBottomRef = useRef(true);
  const thinkingStartRef = useRef<Map<number, number>>(new Map());
  // Guards closeOpenThinking so the O(n) scan doesn't run on every text delta.
  const hasOpenThinkingRef = useRef(false);
  const rafRef = useRef<number | null>(null);

  const refreshSessions = useCallback(() => {
    api.getSessionsWithChildren(30, 0).then((data) => setSessions(data.sessions)).catch(() => {});
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const closeOpenThinking = useCallback((prev: ChatMessage[]): ChatMessage[] => {
    if (!hasOpenThinkingRef.current) return prev;
    for (let i = prev.length - 1; i >= 0; i--) {
      const m = prev[i];
      if (m.type === "thinking" && m.isStreaming) {
        const started = thinkingStartRef.current.get(i);
        const durationMs =
          started != null ? Math.max(0, performance.now() - started) : m.durationMs;
        thinkingStartRef.current.delete(i);
        hasOpenThinkingRef.current = false;
        const updated = [...prev];
        updated[i] = { ...m, isStreaming: false, durationMs };
        return updated;
      }
    }
    hasOpenThinkingRef.current = false;
    return prev;
  }, []);

  const loadSession = useCallback(
    async (sessionId: string) => {
      cancelStream();
      const gen = ++loadGenRef.current;
      setCurrentSessionId(sessionId);
      setIsStreaming(false);
      thinkingStartRef.current.clear();
      hasOpenThinkingRef.current = false;

      try {
        const data = await api.getSessionMessages(sessionId);
        if (loadGenRef.current !== gen) return;

        const chatMsgs: ChatMessage[] = [];
        const toolResults = new Map<string, { content: string; name: string }>();
        for (const m of data.messages) {
          if (m.role === "tool" && m.tool_call_id) {
            toolResults.set(m.tool_call_id, {
              content: typeof m.content === "string" ? m.content : "",
              name: m.tool_name || "",
            });
          }
        }

        for (const m of data.messages) {
          if (m.role === "user") {
            chatMsgs.push({
              type: "user",
              content: typeof m.content === "string" ? m.content : "",
            });
          } else if (m.role === "assistant") {
            if (m.tool_calls) {
              for (const tc of m.tool_calls) {
                const result = toolResults.get(tc.id);
                chatMsgs.push({
                  type: "tool",
                  name: tc.function.name,
                  input: tc.function.arguments || "",
                  output: result?.content || "",
                });
              }
            }
            const text = typeof m.content === "string" ? m.content : "";
            if (text.trim()) {
              chatMsgs.push({ type: "text", content: text });
            }
          }
        }
        setMessages(chatMsgs);
        isAtBottomRef.current = true;
        setShowJumpToLatest(false);
      } catch {
        if (loadGenRef.current === gen) setMessages([]);
      }
    },
    [cancelStream],
  );

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distance < 40;
    isAtBottomRef.current = atBottom;
    const next = !atBottom && isStreaming;
    setShowJumpToLatest((prev) => (prev === next ? prev : next));
  }, [isStreaming]);

  const scrollToBottom = useCallback((smooth = false) => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
    isAtBottomRef.current = true;
    setShowJumpToLatest(false);
  }, []);

  useEffect(() => {
    if (!isAtBottomRef.current) {
      if (isStreaming) setShowJumpToLatest(true);
      return;
    }
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (!el) return;
      el.scrollTop = el.scrollHeight;
    });
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [messages, isStreaming]);

  useEffect(() => {
    if (!isStreaming) {
      inputRef.current?.focus();
      setShowJumpToLatest(false);
    }
  }, [isStreaming]);

  const startNewChat = useCallback(() => {
    cancelStream();
    setCurrentSessionId(null);
    setMessages([]);
    setIsStreaming(false);
    thinkingStartRef.current.clear();
    hasOpenThinkingRef.current = false;
    inputRef.current?.focus();
  }, [cancelStream]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key.toLowerCase() === "k") {
        e.preventDefault();
        startNewChat();
      } else if (meta && e.key === "/") {
        e.preventDefault();
        inputRef.current?.focus();
      } else if (e.key === "Escape") {
        if (isStreaming) {
          cancelStream();
        } else if (document.activeElement === inputRef.current && input) {
          setInput("");
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [startNewChat, cancelStream, isStreaming, input]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    setInput("");
    setIsStreaming(true);
    // New turn — previous scroll-up position is no longer sticky.
    isAtBottomRef.current = true;
    setShowJumpToLatest(false);

    setMessages((prev) => [...prev, { type: "user", content: text }]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      const token = window.__HERMES_SESSION_TOKEN__;
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch("/api/chat", {
        method: "POST",
        headers,
        body: JSON.stringify({ message: text, session_id: currentSessionId }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const errText = await res.text().catch(() => "Unknown error");
        setMessages((prev) => [...prev, { type: "text", content: `Error: ${errText}` }]);
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
                // Close any streaming thinking; append delta to current text message or create one.
                setMessages((prev) => {
                  const withClosed = closeOpenThinking(prev);
                  const last = withClosed[withClosed.length - 1];
                  if (last && last.type === "text") {
                    const updated = [...withClosed];
                    updated[updated.length - 1] = {
                      ...last,
                      content: last.content + (parsed.text || ""),
                    };
                    return updated;
                  }
                  return [...withClosed, { type: "text", content: parsed.text || "" }];
                });
              } else if (currentEvent === "hermes.thinking") {
                setMessages((prev) => {
                  const last = prev[prev.length - 1];
                  if (last && last.type === "thinking" && last.isStreaming) {
                    const updated = [...prev];
                    updated[updated.length - 1] = {
                      ...last,
                      content: last.content + (parsed.text || ""),
                    };
                    return updated;
                  }
                  const idx = prev.length;
                  thinkingStartRef.current.set(idx, performance.now());
                  hasOpenThinkingRef.current = true;
                  return [
                    ...prev,
                    {
                      type: "thinking",
                      content: parsed.text || "",
                      isStreaming: true,
                    },
                  ];
                });
              } else if (currentEvent === "hermes.tool.progress") {
                setMessages((prev) => {
                  const withClosed = closeOpenThinking(prev);
                  return [
                    ...withClosed,
                    {
                      type: "tool",
                      name: parsed.tool,
                      toolCallId: parsed.tool_call_id || undefined,
                      emoji: parsed.emoji,
                      input: "",
                      output: "",
                      isStreaming: true,
                    },
                  ];
                });
              } else if (currentEvent === "hermes.tool.result") {
                setMessages((prev) => {
                  const updated = [...prev];
                  // Match by tool_call_id if available, else most recent streaming card
                  for (let i = updated.length - 1; i >= 0; i--) {
                    const m = updated[i];
                    if (m.type !== "tool") continue;
                    const idMatch = parsed.tool_call_id && m.toolCallId === parsed.tool_call_id;
                    const streamMatch = m.isStreaming;
                    if (idMatch || streamMatch) {
                      updated[i] = {
                        ...m,
                        toolCallId: parsed.tool_call_id || m.toolCallId,
                        input: parsed.input || "",
                        output: parsed.output || "",
                        isStreaming: false,
                      };
                      break;
                    }
                  }
                  return updated;
                });
              } else if (currentEvent === "hermes.tool.completed") {
                setMessages((prev) => {
                  const updated = [...prev];
                  for (let i = updated.length - 1; i >= 0; i--) {
                    const m = updated[i];
                    if (m.type === "tool" && (m.isStreaming || m.name === parsed.tool)) {
                      updated[i] = {
                        ...m,
                        duration: parsed.duration,
                        isError: parsed.is_error,
                        isStreaming: false,
                      };
                      break;
                    }
                  }
                  return updated;
                });
              } else if (currentEvent === "done") {
                setMessages(closeOpenThinking);
                // session_id may have rotated on compression — adopt the canonical one.
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
        setMessages((prev) => [
          ...prev,
          { type: "text", content: `Error: ${(err as Error).message}` },
        ]);
      }
    } finally {
      setMessages(closeOpenThinking);
      setIsStreaming(false);
      abortRef.current = null;
      refreshSessions();
    }
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
          <Button
            onClick={startNewChat}
            variant="outline"
            className="w-full gap-1.5 text-[0.7rem] h-7"
            title="New chat (⌘K)"
          >
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
      <div className="flex-1 flex flex-col min-w-0 relative">
        <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto p-3">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-muted-foreground">
                <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-30" />
                <p className="text-xs">Start a conversation with Hermes</p>
                <p className="mt-2 text-[0.65rem] opacity-70">
                  <kbd className="px-1 py-0.5 border border-border rounded">⌘K</kbd> new chat ·{" "}
                  <kbd className="px-1 py-0.5 border border-border rounded">⌘/</kbd> focus ·{" "}
                  <kbd className="px-1 py-0.5 border border-border rounded">Esc</kbd> cancel
                </p>
              </div>
            </div>
          )}
          {messages.map((msg, i) => {
            if (msg.type === "tool") {
              return <ToolCard key={i} msg={msg} />;
            }
            if (msg.type === "thinking") {
              return (
                <ThinkingBlock
                  key={i}
                  content={msg.content}
                  isStreaming={msg.isStreaming}
                  durationMs={msg.durationMs}
                />
              );
            }
            if (msg.type === "user") {
              return (
                <div key={i} className="flex justify-end mb-3">
                  <div className="group relative max-w-3xl rounded-lg px-3 py-2 bg-primary/10 text-foreground">
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    <CopyButton
                      value={msg.content}
                      className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                    />
                  </div>
                </div>
              );
            }
            // type === "text"
            return (
              <div key={i} className="flex justify-start mb-3">
                <div className="group relative max-w-3xl rounded-lg px-3 py-2 bg-secondary/40 text-foreground">
                  {msg.content ? (
                    <>
                      <Markdown content={msg.content} />
                      <CopyButton
                        value={msg.content}
                        className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                      />
                    </>
                  ) : isStreaming && i === messages.length - 1 ? (
                    <PendingDots />
                  ) : null}
                </div>
              </div>
            );
          })}
          {isStreaming && messages[messages.length - 1]?.type === "user" && (
            <div className="flex justify-start mb-3">
              <div className="rounded-lg px-3 py-2 bg-secondary/40">
                <PendingDots />
              </div>
            </div>
          )}
        </div>

        {/* Jump-to-latest pill */}
        {showJumpToLatest && (
          <button
            onClick={() => scrollToBottom(true)}
            className="absolute left-1/2 -translate-x-1/2 bottom-20 flex items-center gap-1.5 rounded-full border border-border bg-secondary/90 backdrop-blur px-3 py-1.5 text-[0.7rem] text-foreground shadow-md hover:bg-secondary transition-colors cursor-pointer"
          >
            <ArrowDown className="h-3 w-3" /> Jump to latest
          </button>
        )}

        <div className="border-t border-border px-3 py-2">
          <div className="flex gap-2 items-end max-w-3xl mx-auto">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Hermes... (⌘/ to focus)"
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
              <Button
                onClick={() => cancelStream()}
                variant="outline"
                size="sm"
                className="shrink-0 h-10 w-10 p-0"
                title="Cancel (Esc)"
              >
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={sendMessage}
                disabled={!input.trim()}
                size="sm"
                className="shrink-0 h-10 w-10 p-0"
                title="Send (Enter)"
              >
                <Send className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
