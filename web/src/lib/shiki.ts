import type { ThemedToken } from "shiki";
import type { BundledLanguage, BundledTheme, HighlighterGeneric } from "shiki/bundle/web";

type Highlighter = HighlighterGeneric<BundledLanguage, BundledTheme>;

const SUPPORTED_LANGS = [
  "ts",
  "tsx",
  "js",
  "jsx",
  "py",
  "bash",
  "sh",
  "shell",
  "json",
  "yaml",
  "yml",
  "md",
  "markdown",
  "diff",
  "html",
  "css",
] as const;

const THEME = "github-dark-dimmed";

let highlighterPromise: Promise<Highlighter> | null = null;

/**
 * Lazy-load shiki's highlighter with a small bundled language set.
 * Returns a singleton promise so subsequent callers share the same instance.
 */
export function getHighlighter(): Promise<Highlighter> {
  if (highlighterPromise) return highlighterPromise;

  const promise = (async () => {
    const { createHighlighter } = await import("shiki/bundle/web");
    return createHighlighter({
      themes: [THEME],
      langs: [...SUPPORTED_LANGS],
    });
  })();
  promise.catch(() => {
    highlighterPromise = null;
  });
  highlighterPromise = promise;
  return promise;
}

/** Normalize the code-fence lang hint to one shiki knows about. */
export function resolveLang(lang: string | undefined): string | null {
  if (!lang) return null;
  const lower = lang.toLowerCase();
  const aliases: Record<string, string> = {
    typescript: "ts",
    javascript: "js",
    python: "py",
    zsh: "bash",
    shellscript: "bash",
  };
  const resolved = aliases[lower] ?? lower;
  return (SUPPORTED_LANGS as readonly string[]).includes(resolved) ? resolved : null;
}

export interface TokenizedLine {
  tokens: ThemedToken[];
}

/** Tokenize code. Returns null if shiki hasn't loaded yet or lang is unsupported. */
export async function tokenize(
  code: string,
  lang: string,
): Promise<{ lines: TokenizedLine[]; fg: string; bg: string } | null> {
  try {
    const hl = await getHighlighter();
    const result = hl.codeToTokens(code, {
      lang: lang as BundledLanguage,
      theme: THEME,
    });
    return {
      lines: result.tokens.map((tokens) => ({ tokens })),
      fg: result.fg ?? "",
      bg: result.bg ?? "",
    };
  } catch {
    return null;
  }
}

export { THEME as SHIKI_THEME };
