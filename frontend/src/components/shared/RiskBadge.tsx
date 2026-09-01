import type { RiskLevel } from '../../types/index';

interface RiskBadgeProps {
  level: RiskLevel;
}

const LABELS: Record<RiskLevel, string> = {
  high: 'FLAGGED',
  medium: 'WATCH',
  clear: 'CLEAR',
};

export default function RiskBadge({ level }: RiskBadgeProps) {
  const cls = level === 'high' ? 'badge-high' : level === 'medium' ? 'badge-medium' : 'badge-clear';
  return <span className={`badge ${cls}`}>{LABELS[level]}</span>;
}

interface SeverityBadgeProps {
  severity: 'high' | 'medium' | 'neutral';
}

export function SeverityBadge({ severity }: SeverityBadgeProps) {
  const cls = severity === 'high' ? 'badge-high' : severity === 'medium' ? 'badge-medium' : 'badge-neutral';
  const labels = { high: 'HIGH', medium: 'WATCH', neutral: 'INFO' };
  return <span className={`badge ${cls}`}>{labels[severity]}</span>;
}