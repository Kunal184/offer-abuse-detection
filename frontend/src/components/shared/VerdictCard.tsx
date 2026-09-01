import { useEffect, useRef, useState } from 'react';

interface VerdictCardProps {
  number: React.ReactNode;
  statement: string;
  label?: string;
  meta?: string;
  substatement?: string;
}

export default function VerdictCard({ number, statement, label = 'SYSTEM FINDING', meta, substatement }: VerdictCardProps) {
  return (
    <div className="verdict-card">
      <div className="verdict-card-label">{label}</div>
      <div className="verdict-card-number">{number}</div>
      <div className="verdict-card-statement">{statement}</div>
      {substatement && (
        <div className="verdict-card-statement" style={{ marginTop: 4, opacity: 0.7, fontSize: 11 }}>
          {substatement}
        </div>
      )}
      {meta && <div className="verdict-card-meta">{meta}</div>}
    </div>
  );
}

/* ─── Animated count-up number ──────────────────────────────── */

interface AnimatedNumberProps {
  target: number;
  suffix?: string;
  decimals?: number;
  duration?: number;
}

export function AnimatedNumber({ target, suffix = '', decimals = 0, duration = 900 }: AnimatedNumberProps) {
  const [current, setCurrent] = useState(0);
  const startTime = useRef<number | null>(null);
  const frame = useRef<number | null>(null);

  useEffect(() => {
    setCurrent(0);
    startTime.current = null;
    frame.current = null;

    const animate = (timestamp: number) => {
      if (!startTime.current) startTime.current = timestamp;
      const elapsed = timestamp - startTime.current;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setCurrent(eased * target);
      if (progress < 1) {
        frame.current = requestAnimationFrame(animate);
      }
    };

    frame.current = requestAnimationFrame(animate);
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current);
    };
  }, [target, duration]);

  return (
    <span className="count-up">
      {current.toFixed(decimals)}{suffix}
    </span>
  );
}