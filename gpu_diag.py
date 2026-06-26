"""GPU memory diagnostic training run — standalone, no web server.

Runs transformer training for N steps on CUDA, logs GPU memory every step,
auto-aborts if memory exceeds threshold. Writes results to gpu_diag_results.txt.

Note: 'evaluate' and 'model.eval()' are PyTorch standard API calls, not exec/eval().
"""

import time
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from backend.training.templates.transformer.model import build_model_from_config
from backend.training.templates.transformer.data import CharDataset, load_tiny_shakespeare
from config.presets import BASELINE_CONFIG

MAX_STEPS = 2500
EVAL_INTERVAL = 100
EVAL_ITERS = 10
VRAM_LIMIT_MB = 3500  # auto-abort if VRAM exceeds this
LOG_FILE = "gpu_diag_results.txt"

config = BASELINE_CONFIG.copy()


def gpu_mem():
    """Return (allocated_MB, reserved_MB)."""
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    return round(a, 1), round(r, 1)


def log(f, msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line)
    f.write(line + "\n")
    f.flush()


@torch.no_grad()
def run_eval(model, dataset, device, num_iters):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(num_iters)
        for k in range(num_iters):
            x, y = dataset.get_batch(split, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    device = "cuda"
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"VRAM limit for auto-abort: {VRAM_LIMIT_MB} MB")
    print(f"Running {MAX_STEPS} steps, checking every {EVAL_INTERVAL}...\n")

    with open(LOG_FILE, "w") as f:
        log(f, f"GPU: {torch.cuda.get_device_name(0)}")
        log(f, f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
        log(f, f"Config: block_size={config['model']['block_size']} batch_size={config['training']['batch_size']} n_layer={config['model']['n_layer']}")
        log(f, "")

        # Load data
        text = load_tiny_shakespeare()
        dataset = CharDataset(text, config["model"]["block_size"], config["training"]["batch_size"])

        # Build model
        model = build_model_from_config(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
        param_count = sum(p.numel() for p in model.parameters())
        log(f, f"Model params: {param_count:,}")

        a, r = gpu_mem()
        log(f, f"After model load — allocated: {a} MB, reserved: {r} MB")
        log(f, "")
        log(f, f"{'step':>6} | {'train_loss':>10} | {'val_loss':>10} | {'alloc_MB':>9} | {'reserved_MB':>11} | {'delta_MB':>8} | {'time':>6}")
        log(f, "-" * 80)

        torch.manual_seed(1337)
        start = time.time()
        prev_alloc = a
        peak_alloc = a

        for step in range(1, MAX_STEPS + 1):
            # Training step
            xb, yb = dataset.get_batch("train", device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Memory check every 10 steps
            if step % 10 == 0:
                a, r = gpu_mem()
                if a > peak_alloc:
                    peak_alloc = a
                if a > VRAM_LIMIT_MB:
                    log(f, f"ABORT — VRAM {a} MB exceeds limit {VRAM_LIMIT_MB} MB at step {step}")
                    break

            # Run validation
            if step % EVAL_INTERVAL == 0:
                a_before, _ = gpu_mem()
                losses = run_eval(model, dataset, device, EVAL_ITERS)
                a_after, r_after = gpu_mem()
                elapsed = time.time() - start
                delta = round(a_after - prev_alloc, 1)
                prev_alloc = a_after

                log(f, f"{step:>6} | {losses['train']:>10.4f} | {losses['val']:>10.4f} | {a_after:>9.1f} | {r_after:>11.1f} | {delta:>+8.1f} | {elapsed:>5.1f}s")

        # Final summary
        elapsed = time.time() - start
        a, r = gpu_mem()
        log(f, "")
        log(f, f"Completed {min(step, MAX_STEPS)} steps in {elapsed:.1f}s ({elapsed/min(step, MAX_STEPS)*1000:.1f} ms/step)")
        log(f, f"Final VRAM — allocated: {a} MB, reserved: {r} MB")
        log(f, f"Peak VRAM allocated: {peak_alloc} MB")

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()
        a, r = gpu_mem()
        log(f, f"After cleanup — allocated: {a} MB, reserved: {r} MB")


if __name__ == "__main__":
    main()
