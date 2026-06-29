/**
 * MoE-only chart: plots train/val token drop rate over training steps.
 * Drop rate = % of tokens dropped because an expert exceeded its capacity.
 * Healthy training should drive this down over time as the router learns
 * to distribute tokens more evenly across experts.
 */
import MetricChart, { MetricSeries } from "./MetricChart";
import { MetricRow } from "../types";

const SERIES: MetricSeries[] = [
  { key: "train_drop_rate", name: "Train Drop Rate", stroke: "#a78bfa" },
  { key: "val_drop_rate", name: "Val Drop Rate", stroke: "#fbbf24" },
];

interface Props {
  metrics: MetricRow[];
}

export default function DropRateChart({ metrics }: Props) {
  return (
    <MetricChart
      title="Expert Drop Rate (%)"
      metrics={metrics}
      series={SERIES}
      yAxisUnit="%"
      tooltipFormatter={(value: number) => `${value.toFixed(1)}%`}
      shouldRender={(m) => m.some((row) => row.train_drop_rate != null)}
    />
  );
}
