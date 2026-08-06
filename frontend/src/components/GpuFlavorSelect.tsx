// GPU flavor selector (L40S or H100) for serverless GPU runs.
// Only shown when both:
// 1. device is GPU (starts with "cuda")
// 2. backend is serverless ("nebius_endpoint")
// Single source of truth: App state lifts gpuFlavor + setGpuFlavor as props.

interface Props {
  gpuFlavor: string;
  onFlavorChange: (flavor: string) => void;
  disabled?: boolean;
}

export default function GpuFlavorSelect({ gpuFlavor, onFlavorChange, disabled = false }: Props) {
  return (
    <div>
      <label style={{ fontSize: 15, color: "var(--text-dim)", marginRight: 8 }}>
        GPU Flavor
      </label>
      <select
        value={gpuFlavor}
        onChange={(e) => onFlavorChange(e.target.value)}
        disabled={disabled}
        style={{ fontSize: 15, padding: "4px 8px" }}
      >
        <option value="h100">H100 (default)</option>
        <option value="l40s">L40S</option>
      </select>
    </div>
  );
}
