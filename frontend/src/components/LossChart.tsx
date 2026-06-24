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

export default function LossChart({ metrics }: Props) {
  if (metrics.length === 0) {
    return (
      <div className="panel">
        <h3>Loss Curves</h3>
        <div style={{ color: "var(--text-dim)", fontSize: 13, padding: 20, textAlign: "center" }}>
          No metrics yet. Start training to see loss curves.
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3>Loss Curves</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={metrics} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="step"
            stroke="#94a3b8"
            fontSize={11}
            tickFormatter={(v) => String(v)}
          />
          <YAxis stroke="#94a3b8" fontSize={11} />
          <Tooltip
            contentStyle={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 6,
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey="train_loss"
            stroke="#38bdf8"
            strokeWidth={2}
            dot={false}
            name="Train Loss"
          />
          <Line
            type="monotone"
            dataKey="val_loss"
            stroke="#f87171"
            strokeWidth={2}
            dot={false}
            name="Val Loss"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
