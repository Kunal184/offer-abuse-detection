/**
 * Lightweight SVG-based pie/donut chart.
 * Zero third-party dependencies — avoids the recharts requirement
 * that was causing blank pages on Overview and Analytics.
 */

interface Slice {
  name: string;
  value: number;
  color: string;
}

interface MiniPieChartProps {
  data: Slice[];
  size?: number;
  innerRadius?: number;
  outerRadius?: number;
}

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(
  cx: number,
  cy: number,
  innerR: number,
  outerR: number,
  startDeg: number,
  endDeg: number,
): string {
  // Clamp to avoid degenerate arcs
  const span = Math.min(endDeg - startDeg, 359.99);
  const end = startDeg + span;
  const largeArc = span > 180 ? 1 : 0;

  const o1 = polarToCartesian(cx, cy, outerR, startDeg);
  const o2 = polarToCartesian(cx, cy, outerR, end);
  const i1 = polarToCartesian(cx, cy, innerR, end);
  const i2 = polarToCartesian(cx, cy, innerR, startDeg);

  return [
    `M ${o1.x} ${o1.y}`,
    `A ${outerR} ${outerR} 0 ${largeArc} 1 ${o2.x} ${o2.y}`,
    `L ${i1.x} ${i1.y}`,
    `A ${innerR} ${innerR} 0 ${largeArc} 0 ${i2.x} ${i2.y}`,
    'Z',
  ].join(' ');
}

export default function MiniPieChart({
  data,
  size = 120,
  innerRadius = 35,
  outerRadius = 55,
}: MiniPieChartProps) {
  const total = data.reduce((s, d) => s + d.value, 0);
  if (total === 0) return null;

  const cx = size / 2;
  const cy = size / 2;
  let currentDeg = 0;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {data.map((slice) => {
        if (slice.value === 0) return null;
        const sweep = (slice.value / total) * 360;
        const path = describeArc(cx, cy, innerRadius, outerRadius, currentDeg, currentDeg + sweep);
        currentDeg += sweep;
        return <path key={slice.name} d={path} fill={slice.color} />;
      })}
    </svg>
  );
}
