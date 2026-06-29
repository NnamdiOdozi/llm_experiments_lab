import MetricChart, { MetricSeries } from "./MetricChart";
import { MetricRow } from "../types";

const SERIES: MetricSeries[] = [
  { key: "train_loss", name: "Train Loss", stroke: "#38bdf8" },
  { key: "val_loss", name: "Val Loss", stroke: "#f87171" },
];

interface Props {
  metrics: MetricRow[];
}

export default function LossChart({ metrics }: Props) {
  return (
    <MetricChart
      title="Loss Curves"
      metrics={metrics}
      series={SERIES}
      emptyText="No metrics yet. Start training to see loss curves."
    />
  );
}
