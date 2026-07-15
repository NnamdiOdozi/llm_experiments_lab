import { useState, CSSProperties } from "react";

export interface CopyIconButtonProps {
  getText: () => string;
  title: string;
  size?: 12 | 14;
  /** For controlled mode (e.g., ChatPanel's per-message tracking).
   *  When provided, the button uses this prop instead of internal state.
   *  Caller is responsible for timer logic. */
  copied?: boolean;
  /** For controlled mode: called when clipboard write succeeds.
   *  Caller manages the 1.5s timeout. */
  onCopied?: () => void;
}

/**
 * Reusable copy-to-clipboard icon button with checkmark confirmation.
 *
 * Two modes:
 * 1. Uncontrolled (default): self-managed `copied` state, 1.5s auto-timeout.
 *    Used by Inspector.tsx (12px) and App.tsx data tabs (14px).
 * 2. Controlled: `copied` prop + `onCopied` callback for per-message tracking.
 *    Used by ChatPanel.tsx (14px) to track per-message state via copiedId.
 *
 * Icon size is configurable (12px or 14px); defaults to 12px.
 *
 * See docs/DESIGN_DECISIONS.md.
 */
export function CopyIconButton({ getText, title, size = 12, copied: controlledCopied, onCopied }: CopyIconButtonProps) {
  const [internalCopied, setInternalCopied] = useState(false);
  const isControlled = controlledCopied !== undefined;
  const copied = isControlled ? controlledCopied : internalCopied;

  const buttonStyle: CSSProperties = {
    background: "none",
    border: "none",
    color: "var(--text-dim)",
    cursor: "pointer",
    padding: 0,
    lineHeight: 0,
  };

  const sizeStr = size.toString();

  const handleClick = () => {
    navigator.clipboard.writeText(getText());
    if (isControlled) {
      onCopied?.();
    } else {
      setInternalCopied(true);
      setTimeout(() => setInternalCopied(false), 1500);
    }
  };

  return (
    <button
      onClick={handleClick}
      title={title}
      style={buttonStyle}
    >
      {copied ? (
        <svg width={sizeStr} height={sizeStr} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg width={sizeStr} height={sizeStr} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  );
}
