/**
 * MoE-only chart: plots train/val token drop rate over training steps.
 * Drop rate = % of tokens dropped because an expert exceeded its capacity.
 * Healthy training should drive this down over time as the router learns
 * to distribute tokens more evenly across experts.
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

interface Props {
  metrics: MetricRow[];
}

export default function DropRateChart({ metrics }: Props) {
  // Only render if metrics actually contain drop rate data (MoE runs)
  const hasData = metrics.some((m) => m.train_drop_rate != null);
  if (!hasData) return null;

  return (
    <div className="panel">
      <h3>Expert Drop Rate (%)</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={metrics} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="step"
            stroke="#94a3b8"
            fontSize={11}
            tickFormatter={(v) => String(v)}
          />
          <YAxis stroke="#94a3b8" fontSize={11} unit="%" />
          <Tooltip
            contentStyle={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(value: number) => `${value.toFixed(1)}%`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey="train_drop_rate"
            stroke="#a78bfa"
            strokeWidth={2}
            dot={false}
            name="Train Drop Rate"
          />
          <Line
            type="monotone"
            dataKey="val_drop_rate"
            stroke="#fbbf24"
            strokeWidth={2}
            dot={false}
            name="Val Drop Rate"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
