import { useEffect, useRef, useState } from "react";
import { committedNumber } from "../numericDraft";

interface Props {
  value: number;
  min: number;
  max: number;
  step?: number;
  className?: string;
  ariaLabel?: string;
  onCommit: (value: number) => void;
}

export function DraftNumberInput({ value, min, max, step, className, ariaLabel, onCommit }: Props) {
  const [draft, setDraft] = useState(String(value));
  const [editing, setEditing] = useState(false);
  const cancelNextBlur = useRef(false);

  useEffect(() => {
    if (!editing) setDraft(String(value));
  }, [editing, value]);

  function commit() {
    const next = committedNumber(draft, min, max);
    if (next === null) {
      setDraft(String(value));
    } else {
      onCommit(next);
      setDraft(String(next));
    }
    setEditing(false);
  }

  return (
    <input
      aria-label={ariaLabel}
      className={className}
      type="number"
      min={min}
      max={max}
      step={step}
      value={draft}
      onFocus={() => setEditing(true)}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={() => {
        if (cancelNextBlur.current) {
          cancelNextBlur.current = false;
          setDraft(String(value));
          setEditing(false);
          return;
        }
        commit();
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          event.currentTarget.blur();
        }
        if (event.key === "Escape") {
          event.preventDefault();
          cancelNextBlur.current = true;
          setDraft(String(value));
          event.currentTarget.blur();
        }
      }}
    />
  );
}
