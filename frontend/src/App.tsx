import { useState, useEffect, useRef, useCallback, CSSProperties } from "react";
import { ExperimentConfig, MetricRow, RunStatus, ArchitectureNode, DiagnosticSnapshot, TERMINAL_RUN_STATUSES } from "./types";
import PresetPicker from "./components/PresetPicker";
import ExperimentBrowser from "./components/ExperimentBrowser";
import HardwareSpecs from "./components/HardwareSpecs";
import ConfigPanel from "./components/ConfigPanel";
import ArchSchematic from "./components/ArchSchematic";
import Inspector, { SubTab } from "./components/Inspector";
import CodeView from "./components/CodeView";
import LossChart from "./components/LossChart";
import DropRateChart from "./components/DropRateChart";
import TrainingControls from "./components/TrainingControls";
import PausePrompt from "./components/PausePrompt";
import ExportBar from "./components/ExportBar";
import ExperimentNotes from "./components/ExperimentNotes";
import ChatPanel from "./components/ChatPanel";
import WorkerIdleBanner from "./components/WorkerIdleBanner";
import OpenRunsPage from "./components/OpenRunsPage";
import { CopyIconButton } from "./components/CopyIconButton";
import { useActivityHeartbeat } from "./hooks/useActivityHeartbeat";
import { playTrainingStartBeep, isSoundMuted, setSoundMuted, shouldPlayTrainingStartBeep, unlockAudioContext } from "./utils/beep";
import {
  startTraining,
  pauseTraining,
  resumeTraining,
  stopTraining,
  fetchRunStatus,
  fetchMetrics,
  updateConfig,
  fetchExperiment,
  fetchPresets,
  peekDiagnostic,
  ApiError,
  OpenRun,
} from "./hooks/useApi";

const SESSION_KEY = "llm_lab_session";

function saveSession(experimentId: number | null, runId: number | null, config: ExperimentConfig | null) {
  if (experimentId != null && config != null) {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({ experimentId, runId, config }));
  } else {
    sessionStorage.removeItem(SESSION_KEY);
  }
}

function loadSession(): { experimentId: number; runId: number | null; config: ExperimentConfig } | null {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (s.experimentId != null && s.config != null) return s;
  } catch { /* ignore corrupt data */ }
  return null;
}

// "events" reserved (hidden, see §34/§37); dynamic "data-<id>" tabs opened
// by double-clicking a vector, see DataTab below. Widened to string so
// dynamic ids type-check without enumerating every possible id.
type RightPaneTab = string;


const positionTableCellStyle: CSSProperties = {
  border: "1px solid var(--border)",
  padding: "4px 6px",
  textAlign: "left",
  whiteSpace: "nowrap",
};

interface DataTab {
  id: string;
  title: string;
  // Raw values, not a preformatted string — rendered as an index/value
  // dataframe (one row per element), per direct user reference 2026-07-15
  // ("should look like a single column in a data frame... indexes running
  // down"). See docs/DESIGN_DECISIONS.md.
  content: number[];
  // Tab to return to on close — whichever tab was active when this data
  // tab was opened (almost always "inspector"). Previously hardcoded to
  // "assistant", which was jarring since these are always opened from
  // Inspector. See docs/DESIGN_DECISIONS.md.
  returnTo: RightPaneTab;
}

export default function App() {
  const saved = useRef(loadSession());
  const [experimentId, setExperimentId] = useState<number | null>(saved.current?.experimentId ?? null);
  const [config, setConfig] = useState<ExperimentConfig | null>(saved.current?.config ?? null);
  const [runId, setRunId] = useState<number | null>(saved.current?.runId ?? null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [soundMuted, setSoundMutedState] = useState(isSoundMuted);
  // True only between a user's own handleStart() call and that run reaching
  // running/terminal — see beep.ts's top-of-file comment for why this
  // replaced inferring intent from the queued->running status sequence
  // (a fast local run can skip past an observable "queued" reading).
  const awaitingStartRef = useRef(false);

  function toggleSoundMuted() {
    setSoundMutedState((prev) => {
      const next = !prev;
      setSoundMuted(next);
      return next;
    });
  }

  // Beep when training actually starts — real gap, 2026-07-15: serverless
  // endpoints can sit queued for minutes waiting on Nebius to provision,
  // easy to step away and miss the moment it starts. See beep.ts and
  // docs/DESIGN_DECISIONS.md.
  useEffect(() => {
    const current = runStatus?.status ?? null;
    if (shouldPlayTrainingStartBeep(awaitingStartRef.current, current) && !soundMuted) {
      playTrainingStartBeep();
    }
    if (current === "running" || (current != null && TERMINAL_RUN_STATUSES.has(current))) {
      awaitingStartRef.current = false;
    }
  }, [runStatus?.status, soundMuted]);

  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [baselineConfig, setBaselineConfig] = useState<ExperimentConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [device, setDevice] = useState("cpu");
  const [backend, setBackend] = useState("local");
  const [gpuFlavor, setGpuFlavor] = useState("l40s");
  const [showOpenRuns, setShowOpenRuns] = useState(false);
  const [showExperiments, setShowExperiments] = useState(false);
  const [disconnected, setDisconnected] = useState(false);
  const [lastPollSuccess, setLastPollSuccess] = useState<number | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  // Backend rejects max_new_tokens > block_size (see
  // backend/api/experiments.py::update_config) — surfaced here since the
  // debounced PATCH in handleConfigChange previously failed silently
  // (fire-and-forget, no .catch at all). Direct user request, 2026-07-15.
  // See docs/DESIGN_DECISIONS.md.
  const [configError, setConfigError] = useState<string | null>(null);

  // Inspector/diagnostic state
  const [rightPaneTab, setRightPaneTab] = useState<RightPaneTab>("assistant");
  // Double-clicking a vector cell opens one of these — a new, closeable tab
  // with the full vector as static, selectable/copyable text (Colab/VS
  // Code variable-inspector pattern, per direct user reference 2026-07-15).
  // Each double-click opens a separate tab rather than reusing one, per
  // explicit choice — closeable via the × so they don't have to accumulate
  // if unwanted. See docs/DESIGN_DECISIONS.md.
  const [dataTabs, setDataTabs] = useState<DataTab[]>([]);
  function openDataTab(title: string, content: number[]) {
    const id = `data-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setDataTabs((prev) => [...prev, { id, title, content, returnTo: rightPaneTab }]);
    setRightPaneTab(id);
  }
  function closeDataTab(id: string) {
    const tab = dataTabs.find((t) => t.id === id);
    setDataTabs((prev) => prev.filter((t) => t.id !== id));
    setRightPaneTab((current) => (current === id ? tab?.returnTo ?? "assistant" : current));
  }
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<ArchitectureNode | null>(null);
  const [diagnosticSnapshot, setDiagnosticSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);
  // Head picker lives in the Inspector pane (contextual — only shown once an
  // attention node is selected), not the Prompt Model panel. Block is never
  // a separate input at all: it's derived below from whichever attention
  // node is currently selected in the diagram, since selecting that node
  // already says which block you mean. See docs/DESIGN_DECISIONS.md.
  const [attentionHead, setAttentionHead] = useState<number | null>(null);
  const [showQKVDetail, setShowQKVDetail] = useState(false);
  // Lifted here (not local to Inspector) for the same reason as the state
  // above — Inspector unmounts whenever a data tab is open, which would
  // otherwise reset the sub-tab back on close. Direct user report,
  // 2026-07-15. Defaults to "runtime" (not "overview") — direct user
  // request, 2026-07-15: "that's where people are most likely to want to
  // go" when they click a node. See docs/DESIGN_DECISIONS.md.
  const [inspectorActiveTab, setInspectorActiveTab] = useState<SubTab>("runtime");
  // Shifts the attention heatmap/qkv_detail window earlier in the sequence
  // — 0 (default) shows the most recent DIAGNOSTIC_POSITION_WINDOW
  // positions. Real user report, 2026-07-13: the heatmap "gets very busy
  // very quickly" as a session grows, since it previously rendered the
  // *entire* T x T matrix with no cap at all. See docs/DESIGN_DECISIONS.md.
  const [qkvWindowOffset, setAttentionWindowOffset] = useState(0);
  // Same idea as qkvWindowOffset, but for every OTHER node's
  // position_vectors/input_position_vectors (LayerNorm, MLP, embedding,
  // final_norm) — direct user request, 2026-07-15: "a stepper that allows
  // that window to slide backwards in time" for these too, not just
  // attention. See docs/DESIGN_DECISIONS.md.
  const [nodeWindowOffset, setNodeWindowOffset] = useState(0);
  const attentionBlockMatch = selectedNodeId?.match(/^block\.(\d+)\.attention$/);
  const attentionBlock = attentionBlockMatch ? parseInt(attentionBlockMatch[1], 10) : null;
  // Set/cleared by PausePrompt as its diagnostic session starts/ends — used
  // below to auto-refresh attention when Head/Block changes, without
  // requiring a full > click. See docs/DESIGN_DECISIONS.md.
  const [diagnosticSessionId, setDiagnosticSessionId] = useState<string | null>(null);

  // Real bug found 2026-07-13: none of this Inspector/diagnostic selection
  // state was reset when a new run started. Starting run A, then
  // immediately starting run B without reloading, left run A's
  // diagnosticSessionId/attentionBlock/attentionHead alive — the peek
  // effect below then fired immediately against run B using a diagnostic
  // session that was never created for it, right as run B's remote worker
  // was still cold-starting. Confirmed via server log
  // (data/logs/session_2026-07-13_20-40-25.log, 21:32-21:40): a stale
  // diag-b71f2501 session id was reused for both run 158 and run 159's
  // peek calls, both 502ing during the ~7.5min cold start. See
  // docs/DESIGN_DECISIONS.md.
  useEffect(() => {
    setSelectedNodeId(null);
    setSelectedNode(null);
    setAttentionHead(null);
    setShowQKVDetail(false);
    setAttentionWindowOffset(0);
    setNodeWindowOffset(0);
    setDiagnosticSessionId(null);
    setDiagnosticSnapshot(null);
  }, [runId]);

  // Reset the window back to "most recent" whenever a different node is
  // selected — a stale offset from a previously-viewed node would
  // otherwise silently carry over and show the wrong slice.
  useEffect(() => {
    setAttentionWindowOffset(0);
    setNodeWindowOffset(0);
  }, [selectedNodeId]);

  // Real bug report, 2026-07-14: changing Head (or clicking a different
  // block's attention node) only updated local selection — the Inspector
  // kept showing whatever was captured on the last >/>> click, with no
  // indication anything was stale. peekDiagnostic recomputes the current
  // state's snapshot for the newly-selected block/head WITHOUT sampling a
  // new token or advancing the session (backend: run_diagnostic_step_internal,
  // skip_token_generation=True) — so this fires automatically instead of
  // requiring the user to click > again just to see a different head.
  useEffect(() => {
    if (diagnosticSessionId == null || runId == null || attentionBlock == null || attentionHead == null) return;

    // Skip peek when attention_maps covers all current needs:
    // - maps available AND
    // - window offset is 0 (maps not recomputed at offset ≠ 0) AND
    // - (!showQKVDetail OR maps include qkv for all pairs)
    // Otherwise peek: for offset ≠ 0 (need recompute), qkv needed but absent, or maps missing.
    const hasCurrentMaps =
      diagnosticSnapshot?.attention_maps?.available &&
      diagnosticSnapshot.attention_maps.weights &&
      qkvWindowOffset === 0 &&
      (!showQKVDetail || !!diagnosticSnapshot.attention_maps.qkv);

    // The canvas heatmap reads snapshot.attention_full, which is captured for
    // ONE (layer, head) at a time and only when we explicitly ask for it. It's
    // NOT part of attention_maps, so the maps-are-current check above is not
    // enough — we must also already hold the full matrix for the CURRENTLY
    // selected pair, or we have to peek again (attention_full: true below).
    const hasCurrentFull =
      diagnosticSnapshot?.attention_full?.available &&
      diagnosticSnapshot.attention_full.layer === attentionBlock &&
      diagnosticSnapshot.attention_full.head === attentionHead;

    if (hasCurrentMaps && hasCurrentFull) {
      // Data already current (windowed maps/qkv AND the full matrix for this
      // pair), skip the network fetch.
      return;
    }

    let cancelled = false;
    setDiagnosticLoading(true);
    peekDiagnostic(runId, diagnosticSessionId, {
      attention_layer: attentionBlock,
      attention_head: attentionHead,
      qkv_detail: showQKVDetail || undefined,
      attention_window_offset: qkvWindowOffset,
      // Ask for the full T×T matrix for the canvas heatmap (one selected head).
      attention_full: true,
    })
      .then((snapshot) => {
        if (!cancelled) setDiagnosticSnapshot(snapshot);
      })
      .catch((err) => {
        console.error("Peek diagnostic failed:", err);
      })
      .finally(() => {
        if (!cancelled) setDiagnosticLoading(false);
      });
    return () => { cancelled = true; };
  }, [diagnosticSessionId, runId, attentionBlock, attentionHead, showQKVDetail, qkvWindowOffset, diagnosticSnapshot?.attention_maps?.available, diagnosticSnapshot?.attention_full?.available, diagnosticSnapshot?.attention_full?.layer, diagnosticSnapshot?.attention_full?.head]);

  // Same idea as the attention peek effect above, for every OTHER node's
  // position_vectors window (LayerNorm, MLP, embedding, final_norm) —
  // stepping nodeWindowOffset refreshes immediately without requiring a
  // fresh > click, matching the attention stepper's existing UX. Excludes
  // attention nodes (handled by the effect above, which passes its own
  // window offset) and lm_head (no windowed position_vectors there).
  useEffect(() => {
    if (diagnosticSessionId == null || runId == null) return;
    if (selectedNodeId == null || selectedNodeId === "lm_head" || selectedNodeId.includes(".attention")) return;
    let cancelled = false;
    setDiagnosticLoading(true);
    peekDiagnostic(runId, diagnosticSessionId, {
      node_window_offset: nodeWindowOffset,
    })
      .then((snapshot) => {
        if (!cancelled) setDiagnosticSnapshot(snapshot);
      })
      .catch((err) => {
        console.error("Peek diagnostic failed:", err);
      })
      .finally(() => {
        if (!cancelled) setDiagnosticLoading(false);
      });
    return () => { cancelled = true; };
  }, [diagnosticSessionId, runId, selectedNodeId, nodeWindowOffset]);

  const failCountRef = useRef(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const configTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Same condition as WorkerIdleBanner's visibility below — a remote
  // worker's idle clock is only relevant while actually using one.
  // Pass gpuFlavor for GPU runs to select the per-flavor worker session.
  const deviceType = device.startsWith("cuda") ? "gpu" : "cpu";
  const gpuFlavorParam = deviceType === "gpu" ? gpuFlavor : undefined;
  useActivityHeartbeat(device, runStatus?.execution_backend !== "local", gpuFlavorParam);

  function handleConfigChange(cfg: ExperimentConfig) {
    setConfig(cfg);
    setConfigError(null);
    if (configTimerRef.current) clearTimeout(configTimerRef.current);
    if (experimentId != null) {
      configTimerRef.current = setTimeout(() => {
        updateConfig(experimentId, cfg).catch((err) => {
          // Previously fire-and-forget — a rejected PATCH (e.g.
          // max_new_tokens > block_size) failed completely silently, the
          // invalid value stayed shown as if it had saved. See
          // docs/DESIGN_DECISIONS.md.
          setConfigError(err instanceof Error ? err.message : String(err));
        });
      }, 500);
    }
  }

  // Inspector/data-tab state is per-run — carrying it across an experiment
  // or run switch left stale vector tabs open and a dangling rightPaneTab
  // pointing at a data-tab id that no longer exists (blank Inspector pane).
  // Direct user report, 2026-07-16. See docs/DESIGN_DECISIONS.md.
  function resetInspectorState() {
    setDataTabs([]);
    setRightPaneTab("assistant");
    setSelectedNode(null);
    setSelectedNodeId(null);
    setDiagnosticSnapshot(null);
    setDiagnosticSessionId(null);
  }

  function handlePresetSelect(expId: number, cfg: ExperimentConfig, selectedDevice: string, selectedBackend: string, selectedGpuFlavor: string) {
    setExperimentId(expId);
    setConfig(cfg);
    setRunId(null);
    setRunStatus(null);
    setMetrics([]);
    resetInspectorState();
    setDevice(selectedDevice);
    setBackend(selectedBackend);
    setGpuFlavor(selectedGpuFlavor);
    saveSession(expId, null, cfg);
  }

  // Reopening a past experiment (which may already have runs) to add a new
  // one — mirrors handlePresetSelect but skips creating a new experiment.
  // Device/backend/gpuFlavor reset to the same defaults a fresh session starts with;
  // TrainingControls lets the user change them before clicking Start.
  function handleLoadExperiment(expId: number, cfg: ExperimentConfig) {
    setExperimentId(expId);
    setConfig(cfg);
    setRunId(null);
    setRunStatus(null);
    setMetrics([]);
    resetInspectorState();
    setDevice("cpu");
    setBackend("local");
    setGpuFlavor("l40s");
    saveSession(expId, null, cfg);
  }

  // Reopening a specific run (paused/running) from Open Runs — previously
  // that page could only Stop a run, with no way back into it at all.
  // Direct user report, 2026-07-15. See docs/DESIGN_DECISIONS.md.
  async function handleReopenRun(run: OpenRun) {
    const exp = await fetchExperiment(run.experiment_id);
    setExperimentId(run.experiment_id);
    setConfig(exp.config);
    setRunId(run.id);
    setRunStatus(null);
    setMetrics([]);
    resetInspectorState();
    setDevice(run.device);
    setBackend(run.execution_backend);
    // GPU flavor would come from the run's data if stored; for now, default to l40s
    // (backward compat: runs created before this feature won't have it set)
    setGpuFlavor("l40s");
    saveSession(run.experiment_id, run.id, exp.config);
    setShowOpenRuns(false);
  }

  const pollStatus = useCallback(async () => {
    if (runId == null) return;
    try {
      const status = await fetchRunStatus(runId);
      setRunStatus(status);
      // Real bug, 2026-07-15: device defaulted to hardcoded "cpu" and was
      // never re-synced from the server, so HardwareSpecs showed CPU-only
      // specs after a reload even when the actual connected run was GPU.
      // Same bug class as the earlier config-staleness fix. See
      // docs/DESIGN_DECISIONS.md.
      if (status.device) setDevice(status.device);
      // Same bug, same fix, for backend — direct user report, 2026-07-16:
      // a serverless run showed "Local" in TrainingControls' Device/Backend
      // picker once it finished. `backend` state defaults to "local" on
      // mount and isn't in sessionStorage, so a refresh mid-run left it
      // stuck at that default forever (the run's own status/badge stayed
      // correct throughout — only this picker, which the isDone block
      // reveals once the run finishes, was ever wrong). See
      // docs/DESIGN_DECISIONS.md.
      if (status.execution_backend) setBackend(status.execution_backend);
      const m = await fetchMetrics(runId);
      setMetrics(m);
      failCountRef.current = 0;
      setDisconnected(false);
      setLastPollSuccess(Date.now());
      setPollError(null);
      if (status.status === "completed" || status.status === "failed" || status.status === "cancelled") {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch (err) {
      // An ApiError means the backend ANSWERED (4xx/5xx) — connected, just
      // unhappy about this request. Only genuine fetch failures (TypeError
      // etc.) count toward the disconnected banner. Real bug, 2026-07-14:
      // the previous check matched the message text against /^4\d\d/,
      // which broke the day api() started throwing FastAPI's detail
      // string instead of "404 Not Found" — a stale sessionStorage runId
      // after a backend restart produced "Run not found" 4xxs that were
      // misread as network failures, showing a persistent false
      // "Backend disconnected" banner. See docs/DESIGN_DECISIONS.md.
      const isNetworkError = !(err instanceof ApiError);
      if (isNetworkError) {
        failCountRef.current += 1;
        if (failCountRef.current >= 3) setDisconnected(true);
      }
      setPollError(err instanceof Error ? err.message : "Poll failed");
    }
  }, [runId]);

  useEffect(() => {
    if (runId == null) return;
    pollStatus();
    pollRef.current = setInterval(pollStatus, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [runId, pollStatus]);

  // Diff-from-baseline: every experiment is created from a preset
  // (PresetPicker is the only creation path), so preset_key is always set.
  // Look it up to show the original values as shadow text in ConfigPanel.
  //
  // This is also the only place `config` ever gets refreshed from the
  // server after initial load. Real bug found via a full API trace,
  // 2026-07-14: `config` state is seeded once from sessionStorage (which
  // survives a hard refresh — only cleared by closing the tab), and this
  // effect used to fetch the live experiment only to compute the baseline,
  // throwing away `exp.config` instead of using it to correct a stale
  // cached copy. Confirmed live: max_new_tokens stayed frozen at a stale
  // value (50) across a hard refresh AND a backend restart, while the
  // Config panel showed the real, current server value (100) the whole
  // time — because ConfigPanel computes its own display default
  // separately and was never the actual source of truth being sent on >>.
  // Now every experiment load re-syncs config to the server's real value.
  // See docs/DESIGN_DECISIONS.md.
  useEffect(() => {
    if (experimentId == null) {
      setBaselineConfig(null);
      return;
    }
    let cancelled = false;
    (async () => {
      const [exp, presets] = await Promise.all([fetchExperiment(experimentId), fetchPresets()]);
      if (cancelled) return;
      setConfig(exp.config);
      const preset = presets.find((p) => p.key === exp.preset_key);
      setBaselineConfig(
        preset ? { template: preset.template, model: preset.model, training: preset.training, inference: preset.inference } : null,
      );
    })();
    return () => { cancelled = true; };
  }, [experimentId]);

  async function handleStart() {
    if (experimentId == null || !config) return;
    // Must run synchronously here, inside the click's own call stack — the
    // browser's autoplay policy can permanently suspend an AudioContext
    // created later (e.g. inside a polling effect), even for a real user-
    // triggered event several ticks removed from the original gesture.
    unlockAudioContext();
    awaitingStartRef.current = true;
    setLoading(true);
    setStartError(null);
    try {
      // Flush any pending config debounce before starting
      if (configTimerRef.current) {
        clearTimeout(configTimerRef.current);
        configTimerRef.current = null;
        await updateConfig(experimentId, config);
      }
      const { run_id } = await startTraining(experimentId, device, backend, gpuFlavor);
      setRunId(run_id);
      saveSession(experimentId, run_id, config);
    } catch (err) {
      awaitingStartRef.current = false; // start never actually happened
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.startsWith("429")) {
        setStartError("Max concurrent runs reached. Stop a run first.");
      } else {
        setStartError(msg);
      }
    }
    setLoading(false);
  }

  async function handlePause() {
    if (runId == null) return;
    setLoading(true);
    setControlError(null);
    try {
      await pauseTraining(runId);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : "Pause failed");
    }
    setLoading(false);
  }

  async function handleResume() {
    if (runId == null) return;
    setLoading(true);
    setControlError(null);
    try {
      await resumeTraining(runId);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : "Resume failed");
    }
    setLoading(false);
  }

  // A stopped run is genuinely dead — no resume, no prompt (unlike paused,
  // which is what actually stays useful) — so lingering step/elapsed
  // numbers just look like stale "still going" state rather than the
  // finished run it is. Reset to the same fresh, ready-to-start state as
  // picking a preset — same experiment/config, no run. The stopped run's
  // own final numbers stay browsable via Open Runs, nothing is lost.
  // Direct user request, 2026-07-15. See docs/DESIGN_DECISIONS.md.
  function resetAfterStop() {
    setRunId(null);
    setRunStatus(null);
    setMetrics([]);
    resetInspectorState();
    saveSession(experimentId, null, config);
  }

  async function handleStop() {
    if (runId == null) return;
    setLoading(true);
    setControlError(null);
    try {
      await stopTraining(runId);
      resetAfterStop();
    } catch (err) {
      // Backend's /stop route only ever raises 400 for one reason — no
      // active process/session to stop, i.e. the run was already dead
      // (completed/failed/cancelled, or a stale runId from a previous
      // session). Real bug, 2026-07-15: this was treated as a fresh
      // failure — persistent red "Run not found" banner, no cleanup —
      // even though the user's actual goal (stop the run) was already
      // true. Idempotent: stopping an already-stopped run should succeed
      // quietly, not error. Any OTHER status code (network error, 5xx)
      // still surfaces as a real error below.
      if (err instanceof ApiError && err.status === 400) {
        resetAfterStop();
      } else {
        setControlError(err instanceof Error ? err.message : "Stop failed");
      }
    }
    setLoading(false);
  }

  if (showOpenRuns) {
    return <OpenRunsPage onClose={() => setShowOpenRuns(false)} onReopen={handleReopenRun} />;
  }

  if (showExperiments) {
    return (
      <div style={{ maxWidth: 700, margin: "40px auto", padding: "0 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h1 style={{ fontSize: 20 }}>Existing Experiments</h1>
          <button onClick={() => setShowExperiments(false)}>← Back</button>
        </div>
        <ExperimentBrowser
          onSelect={(expId, cfg) => { handleLoadExperiment(expId, cfg); setShowExperiments(false); }}
        />
      </div>
    );
  }

  // No experiment selected: show preset picker. Bumped from 700 to 900 to
  // give the ~50% bigger preset grid (see PresetPicker.tsx) room to
  // breathe. HardwareSpecs (unlabeled "Serverless CPU/GPU: ... (live)"
  // line) removed from here — it read as a live-status indicator but
  // wasn't one (both entries always showed "live" together regardless of
  // which worker, if any, was actually running), so it was just confusing
  // rather than informative. Direct user reports, 2026-07-15. See
  // docs/DESIGN_DECISIONS.md. Existing-experiments browsing moved behind
  // its own page (matching the existing Open Runs pattern) instead of
  // always-inline, to keep this landing view uncluttered.
  if (!experimentId || !config) {
    return (
      <div style={{ maxWidth: 900, margin: "60px auto", padding: "0 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <h1 style={{ fontSize: 24 }}>LLM Experiments Lab</h1>
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => setShowExperiments(true)}>Existing Experiments</button>
            <button onClick={() => setShowOpenRuns(true)}>Open Runs</button>
          </div>
        </div>
        <p style={{ color: "var(--text-dim)", marginBottom: 20, fontSize: 14 }}>
          Pick a preset to create an experiment. Tweak the config, train, and watch loss curves.
        </p>
        <PresetPicker onSelect={handlePresetSelect} />
      </div>
    );
  }

  // Experiment selected: show lab workspace
  return (
    <div style={{ padding: 20 }}>
      {disconnected && (
        <div style={{
          background: "var(--red, #e53e3e)",
          color: "#fff",
          padding: "8px 16px",
          borderRadius: 6,
          marginBottom: 12,
          fontSize: 13,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}>
          <span>⚠ Backend disconnected — restart the server and refresh</span>
          <button
            style={{ background: "rgba(255,255,255,0.2)", border: "none", color: "#fff", padding: "4px 10px", borderRadius: 4, cursor: "pointer" }}
            onClick={() => { setDisconnected(false); failCountRef.current = 0; }}
          >
            Dismiss
          </button>
        </div>
      )}
      {/* A remote run sits at "queued" for as long as the endpoint takes to
          come up (cold GPU restarts can take several minutes) — this is
          expected, not a failure, but looks identical to a real outage if
          left unexplained. See docs/DESIGN_DECISIONS.md. */}
      {runStatus?.status === "queued" && runStatus?.execution_backend !== "local" && (
        <div style={{
          background: "var(--accent-dim)",
          color: "#fff",
          padding: "8px 16px",
          borderRadius: 6,
          marginBottom: 12,
          fontSize: 13,
        }}>
          Waiting for the serverless {device === "cuda" ? "GPU" : "CPU"} endpoint to start —
          {device === "cuda"
            ? " a cold GPU restart can take up to ~5 minutes."
            : " a cold CPU restart is usually faster, up to ~2 minutes."}{" "}
          Training starts automatically once it's ready, no action needed.
        </div>
      )}
      {/* Hide when the current run is definitively local — a remote worker's
          idle status is irrelevant noise if you're not using it right now.
          See docs/DESIGN_DECISIONS.md §10. */}
      {runStatus?.execution_backend !== "local" && <WorkerIdleBanner device={device} gpuFlavor={deviceType === "gpu" ? gpuFlavor : undefined} />}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <h1 style={{ fontSize: 20 }}>
          LLM Experiments Lab
          <span style={{ color: "var(--text-dim)", fontSize: 14, marginLeft: 12 }}>
            Experiment #{experimentId}
          </span>
        </h1>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setShowOpenRuns(true)}>Open Runs</button>
          <button onClick={() => { setExperimentId(null); setConfig(null); setRunId(null); setRunStatus(null); setMetrics([]); resetInspectorState(); saveSession(null, null, null); }}>
            ← New Experiment
          </button>
        </div>
      </div>
      <div style={{ marginBottom: 12 }}>
        {/* runStatus.execution_backend (the active run's real backend) takes
            priority over the device/backend picker state, which is only a
            pending choice for the *next* Start click. */}
        <HardwareSpecs device={device} backend={runStatus?.execution_backend ?? backend} />
      </div>

      {/* Dashboard columns. Layout (fluid widths + shrink-then-stack
          breakpoint) lives in index.css .dashboard-grid so a @media query can
          reflow it for narrow screens / an open browser side-pane — inline
          styles can't express @media. Column proportions (left ~456, right
          ~684, direct user request 2026-07-16) preserved as the max track
          sizes. See docs/DESIGN_DECISIONS.md. */}
      <div className="dashboard-grid">
        {/* Left sidebar */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <ConfigPanel
            config={config}
            onChange={handleConfigChange}
            disabled={runStatus?.status === "running"}
            baseline={baselineConfig}
            error={configError}
          />
          <TrainingControls
            runId={runId}
            runStatus={runStatus}
            onStart={handleStart}
            onPause={handlePause}
            onResume={handleResume}
            onStop={handleStop}
            loading={loading}
            device={device}
            onDeviceChange={setDevice}
            backend={backend}
            onBackendChange={setBackend}
            gpuFlavor={gpuFlavor}
            onGpuFlavorChange={setGpuFlavor}
            lastPollSuccess={lastPollSuccess}
            pollError={pollError}
            startError={startError}
            controlError={controlError}
            soundMuted={soundMuted}
            onToggleSoundMuted={toggleSoundMuted}
          />
          <ExportBar experimentId={experimentId} runId={runId} />
          <ExperimentNotes experimentId={experimentId} />
        </div>

        {/* Main area */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Loss + MoE drop rate charts side-by-side when MoE data present */}
          <div style={{ display: "flex", gap: 16 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <LossChart metrics={metrics} />
            </div>
            {metrics.some((m) => m.train_drop_rate != null) && (
              <div style={{ flex: 1, minWidth: 0 }}>
                <DropRateChart metrics={metrics} />
              </div>
            )}
          </div>
          <ArchSchematic
            runId={runId}
            config={config}
            onNodeClick={(nodeId, node) => {
              setSelectedNodeId(nodeId);
              setSelectedNode(node);
              setRightPaneTab("inspector");
            }}
            selectedNodeId={selectedNodeId}
          />
          {runId != null && (
            <PausePrompt
              runId={runId}
              // Was paused||completed only — a local run's checkpoint
              // persists on the controller's own disk regardless of
              // terminal status (periodic saves during training, not just
              // at the end), so a reopened local failed/cancelled run can
              // still be prompted too. A remote run's checkpoint dies with
              // its container either way — the backend already handles
              // that gracefully (HTTPException(400, "No checkpoint
              // available for this run")) rather than needing a frontend-
              // side check here. Direct user request, 2026-07-15. See
              // docs/DESIGN_DECISIONS.md §79b.
              canPrompt={runStatus?.status === "paused" || (runStatus != null && TERMINAL_RUN_STATUSES.has(runStatus.status))}
              attentionBlock={attentionBlock}
              attentionHead={attentionHead}
              showQKVDetail={showQKVDetail}
              attentionWindowOffset={qkvWindowOffset}
              nodeWindowOffset={nodeWindowOffset}
              // Same config.inference.max_new_tokens the ConfigPanel already
              // shows and Generate already uses (server-side, via
              // prompt_paused_model's inference_cfg.get(...)) — >> was
              // hardcoding 50 regardless of this value. See
              // docs/DESIGN_DECISIONS.md.
              maxNewTokens={typeof config?.inference?.max_new_tokens === "number" ? config.inference.max_new_tokens : 50}
              temperature={typeof config?.inference?.temperature === "number" ? config.inference.temperature : 0.8}
              decodingMode={typeof config?.inference?.decoding_mode === "string" ? config.inference.decoding_mode : "sample"}
              onDiagnosticSnapshot={(snapshot) => {
                setDiagnosticSnapshot(snapshot);
                setDiagnosticLoading(false);
              }}
              onSessionIdChange={setDiagnosticSessionId}
            />
          )}
          <CodeView experimentId={experimentId} runId={runId} />
        </div>

        {/* Right pane: Lab Assistant / Inspector / Events, sticky, runs top-to-bottom of the viewport */}
        <div style={{ position: "sticky", top: 20, height: "calc(100vh - 100px)" }}>
          {/* Tabs header */}
          <div
            style={{
              display: "flex",
              gap: 24,
              borderBottom: "1px solid var(--border)",
              marginBottom: 12,
              backgroundColor: "var(--surface)",
              borderRadius: "8px 8px 0 0",
              padding: "0 16px",
            }}
          >
            {/* "events" tab hidden for now (2026-07-14) — it only ever showed
                "Coming soon" (see docs/Diagnostic_Contract.md's Outstanding
                section), which read as confusing/broken rather than deferred.
                Re-enable by adding "events" back here once it has real
                content. See docs/DESIGN_DECISIONS.md. */}
            {["assistant", "inspector"].map((tab) => (
              <button
                key={tab}
                onClick={() => setRightPaneTab(tab)}
                style={{
                  background: "none",
                  border: "none",
                  color: rightPaneTab === tab ? "var(--accent)" : "var(--text-dim)",
                  cursor: "pointer",
                  padding: "12px 0",
                  fontSize: 12,
                  fontWeight: rightPaneTab === tab ? 600 : 400,
                  borderBottom: rightPaneTab === tab ? "2px solid var(--accent)" : "none",
                  transition: "all 0.15s",
                  textTransform: "capitalize",
                }}
              >
                {tab}
              </button>
            ))}
            {dataTabs.map((t) => (
              <div key={t.id} style={{ display: "flex", alignItems: "center", gap: 2 }}>
                <button
                  onClick={() => setRightPaneTab(t.id)}
                  style={{
                    background: "none",
                    border: "none",
                    color: rightPaneTab === t.id ? "var(--accent)" : "var(--text-dim)",
                    cursor: "pointer",
                    padding: "12px 0",
                    fontSize: 12,
                    fontWeight: rightPaneTab === t.id ? 600 : 400,
                    borderBottom: rightPaneTab === t.id ? "2px solid var(--accent)" : "none",
                    transition: "all 0.15s",
                    whiteSpace: "nowrap",
                  }}
                >
                  {t.title}
                </button>
                <button
                  onClick={() => closeDataTab(t.id)}
                  title="Close"
                  style={{ background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer", fontSize: 12, padding: "0 4px" }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          {/* Tab content */}
          <div style={{ height: "calc(100% - 50px)", overflowY: "auto" }}>
            {rightPaneTab === "assistant" && <ChatPanel experimentId={experimentId} />}
            {rightPaneTab === "inspector" && (
              <Inspector
                runId={runId}
                selectedNode={selectedNode}
                selectedNodeId={selectedNodeId}
                diagnosticSnapshot={diagnosticSnapshot}
                currentStep={diagnosticSnapshot?.generation_step ?? null}
                isLoading={diagnosticLoading}
                attentionHead={attentionHead}
                onAttentionHeadChange={setAttentionHead}
                showQKVDetail={showQKVDetail}
                onShowQKVDetailChange={setShowQKVDetail}
                qkvWindowOffset={qkvWindowOffset}
                onQkvWindowOffsetChange={setAttentionWindowOffset}
                nodeWindowOffset={nodeWindowOffset}
                onNodeWindowOffsetChange={setNodeWindowOffset}
                numHeads={typeof config?.model?.n_head === "number" ? config.model.n_head : null}
                onOpenDataTab={openDataTab}
                activeTab={inspectorActiveTab}
                onActiveTabChange={setInspectorActiveTab}
              />
            )}
            {/* {rightPaneTab === "events" && (
              <div className="panel">
                <h3>Events</h3>
                <p style={{ fontSize: 12, color: "var(--text-dim)" }}>Event log coming soon in Phase 2.</p>
              </div>
            )} */}
            {dataTabs.map((t) => rightPaneTab === t.id && (
              <div key={t.id} className="panel">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <h3 style={{ margin: 0 }}>{t.title}</h3>
                  <div style={{ display: "flex", gap: 8 }}>
                    {/* Full-precision values (t.content itself, not the
                        .toFixed(6) truncated display below) — one per line,
                        matching this tab's single-column layout, pastes as
                        a single spreadsheet column. Direct user request,
                        2026-07-15. See docs/DESIGN_DECISIONS.md. */}
                    <CopyIconButton size={14} getText={() => t.content.join("\n")} title="Copy full vector" />
                    <button onClick={() => closeDataTab(t.id)}>Close</button>
                  </div>
                </div>
                {/* Single-column dataframe/series style — index running down
                    the left, one value per row — not a flat bracketed
                    string. Direct user reference 2026-07-15: "should look
                    like a single column in a data frame... indexes running
                    down." Still plain selectable/copyable text per cell,
                    same as Colab/VS Code's variable inspector. See
                    docs/DESIGN_DECISIONS.md. */}
                <div style={{ height: "calc(100vh - 260px)", overflowY: "auto", border: "1px solid var(--border)", borderRadius: 4 }}>
                  {/* No width: "100%" — this is a 2-column Index/Value table
                      of short numbers; stretching it to the full ~570px
                      pane left huge empty gaps in each cell. Fixed width
                      here (roughly 2x the auto-content width) rather than
                      100% or auto — direct user report, 2026-07-16: auto
                      was too narrow, 100% was too wide. See
                      docs/DESIGN_DECISIONS.md. */}
                  <table style={{ borderCollapse: "collapse", fontSize: 10, fontFamily: "var(--font-mono)", width: 240 }}>
                    <thead>
                      <tr>
                        <th style={{ ...positionTableCellStyle, position: "sticky", top: 0, background: "var(--surface)" }}>Index</th>
                        <th style={{ ...positionTableCellStyle, position: "sticky", top: 0, background: "var(--surface)" }}>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {t.content.map((v, i) => (
                        <tr key={i}>
                          <td style={{ ...positionTableCellStyle, color: "var(--text-dim)" }}>{i}</td>
                          <td style={positionTableCellStyle}>{v.toFixed(6)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
