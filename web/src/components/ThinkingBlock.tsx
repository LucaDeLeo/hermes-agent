import { memo, useEffect, useRef, useState } from "react";
import { Brain, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { Markdown } from "@/components/Markdown";

function ThinkingBlockImpl({
  content,
  isStreaming,
  durationMs,
}: {
  content: string;
  isStreaming?: boolean;
  durationMs?: number;
}) {
  const [expanded, setExpanded] = useState<boolean>(Boolean(isStreaming));
  const prevStreamingRef = useRef(isStreaming);

  useEffect(() => {
    if (prevStreamingRef.current && !isStreaming) setExpanded(false);
    prevStreamingRef.current = isStreaming;
  }, [isStreaming]);

  const durationLabel =
    durationMs != null ? `${(durationMs / 1000).toFixed(1)}s` : null;

  return (
    <div className="my-2 max-w-3xl mx-auto">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[0.7rem] text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
      >
        {expanded ? (
          <ChevronDown className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0" />
        )}
        <Brain className="h-3 w-3 shrink-0 opacity-70" />
        {isStreaming ? (
          <span className="flex items-center gap-1.5">
            <span className="italic">Thinking</span>
            <span className="inline-flex gap-0.5">
              <span className="h-1 w-1 rounded-full bg-current opacity-60 animate-[pulse_1.2s_ease-in-out_infinite]" />
              <span className="h-1 w-1 rounded-full bg-current opacity-60 animate-[pulse_1.2s_ease-in-out_0.2s_infinite]" />
              <span className="h-1 w-1 rounded-full bg-current opacity-60 animate-[pulse_1.2s_ease-in-out_0.4s_infinite]" />
            </span>
            <Loader2 className="h-3 w-3 animate-spin opacity-70" />
          </span>
        ) : (
          <span>
            Thought{durationLabel ? ` for ${durationLabel}` : ""}
          </span>
        )}
      </button>
      {expanded && content && (
        <div className="mt-1 ml-4 border-l-2 border-primary/20 pl-3 text-muted-foreground italic text-[0.78rem] leading-relaxed whitespace-pre-wrap">
          <Markdown content={content} />
        </div>
      )}
    </div>
  );
}

export const ThinkingBlock = memo(ThinkingBlockImpl);
