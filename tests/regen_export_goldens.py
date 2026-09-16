"""Regenerate the export golden files. Run only when a change is intended.

    .venv/bin/python -m tests.regen_export_goldens
"""

import json
from pathlib import Path

from backend.export import build_script

FIXTURES = Path(__file__).parent / "fixtures" / "export"

for config_path in sorted(FIXTURES.glob("*_config.json")):
    key = config_path.name.removesuffix("_config.json")
    config = json.loads(config_path.read_text())
    target = FIXTURES / f"{key}_script.py.golden"
    target.write_text(build_script(config))
    print(f"regenerated {target}")
