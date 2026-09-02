/**
 * Lightweight SVG horizontal bar chart.
 * Replaces recharts BarChart — zero extra dependencies.
 */

interface BarDatum {
  name: string;
  value: number;
  color: string;
}

interface SimpleBarChartProps {
  data: BarDatum[];
  height?: number;
  domain?: [number, number];
}

export default function SimpleBarChart({
  data,
  height = 200,
  domain = [0, 1],
}: SimpleBarChartProps) {
  const [min, max] = domain;
  const range = max - min || 1;

  const rowHeight = Math.floor(height / data.length);
  const barHeight = Math.max(Math.floor(rowHeight * 0.55), 10);
  const labelWidth = 80;
  const valueWidth = 54;
  const gap = 8;
  const barAreaWidth = 260;
  const svgWidth = labelWidth + gap + barAreaWidth + gap + valueWidth;
  const svgHeight = data.length * rowHeight;

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
      style={{ display: 'block' }}
    >
      {data.map((d, i) => {
        const barW = Math.max(((d.value - min) / range) * barAreaWidth, 2);
        const y = i * rowHeight;
        const barY = y + (rowHeight - barHeight) / 2;

        return (
          <g key={d.name}>
            {/* Label */}
            <text
              x={labelWidth}
              y={y + rowHeight / 2 + 4}
              textAnchor="end"
              fontSize={11}
              fill="#8A8A88"
              fontFamily="inherit"
            >
              {d.name}
            </text>
            {/* Bar background */}
            <rect
              x={labelWidth + gap}
              y={barY}
              width={barAreaWidth}
              height={barHeight}
              rx={3}
              fill="#242422"
            />
            {/* Bar fill */}
            <rect
              x={labelWidth + gap}
              y={barY}
              width={barW}
              height={barHeight}
              rx={3}
              fill={d.color}
            />
            {/* Value */}
            <text
              x={labelWidth + gap + barAreaWidth + gap}
              y={y + rowHeight / 2 + 4}
              fontSize={11}
              fill="#9BA3AB"
              fontFamily="'JetBrains Mono', monospace"
            >
              {d.value.toFixed(4)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
