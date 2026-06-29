/**
 * Generic Recharts line chart — reusable for loss, drop rate, or any future metric.
 * Eliminates duplicated Recharts scaffolding across chart components.
 */
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { MetricRow } from "../types";

export interface MetricSeries {
  key: keyof MetricRow;
  name: string;
  stroke: string;
}

interface Props {
  title: string;
  metrics: MetricRow[];
  series: MetricSeries[];
  emptyText?: string;
  yAxisUnit?: string;
  tooltipFormatter?: (value: number) => string;
  /** If provided, chart returns null when this returns false */
  shouldRender?: (metrics: MetricRow[]) => boolean;
}

export default function MetricChart({
  title,
  metrics,
  series,
  emptyText = "No data yet.",
  yAxisUnit,
  tooltipFormatter,
  shouldRender,
}: Props) {
  if (shouldRender && !shouldRender(metrics)) return null;

  if (metrics.length === 0) {
    return (
      <div className="panel">
        <h3>{title}</h3>
        <div style={{ color: "var(--text-dim)", fontSize: 13, padding: 20, textAlign: "center" }}>
          {emptyText}
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={metrics} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="step"
            stroke="#94a3b8"
            fontSize={11}
            tickFormatter={(v) => String(v)}
          />
          <YAxis stroke="#94a3b8" fontSize={11} unit={yAxisUnit} />
          <Tooltip
            contentStyle={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={tooltipFormatter}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {series.map((s) => (
            <Line
              key={String(s.key)}
              type="monotone"
              dataKey={String(s.key)}
              stroke={s.stroke}
              strokeWidth={2}
              dot={false}
              name={s.name}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
