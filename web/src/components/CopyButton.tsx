import { memo, useState, useCallback } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

function CopyButtonImpl({
  value,
  className,
  label,
}: {
  value: string;
  className?: string;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);

  const onCopy = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      try {
        await navigator.clipboard.writeText(value);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      } catch {
        // clipboard blocked (insecure context) — ignore
      }
    },
    [value],
  );

  return (
    <button
      type="button"
      onClick={onCopy}
      title={copied ? "Copied" : label || "Copy"}
      aria-label={copied ? "Copied" : label || "Copy"}
      className={cn(
        "inline-flex items-center gap-1 rounded-md border border-border bg-secondary/60 px-1.5 py-1 text-[0.65rem] text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors cursor-pointer",
        className,
      )}
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
    </button>
  );
}

export const CopyButton = memo(CopyButtonImpl);
