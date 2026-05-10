import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = {
    "sft": ROOT / "data/sft/sft_seed.jsonl",
    "dpo": ROOT / "data/preference/dpo_seed.jsonl",
    "eval": ROOT / "data/eval/eval_dev.jsonl",
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {line_no} JSON parse failed: {e}") from e
    return rows


def check_image_paths(obj, missing):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"image", "image_path"} and isinstance(value, str):
                path = ROOT / value
                if not path.exists():
                    missing.append(value)
            else:
                check_image_paths(value, missing)
    elif isinstance(obj, list):
        for item in obj:
            check_image_paths(item, missing)


def main():
    has_failure = False

    for name, path in FILES.items():
        print(f"\nChecking {name}: {path}")

        if not path.exists():
            print(f"[FAIL] Missing file: {path}")
            has_failure = True
            continue

        rows = read_jsonl(path)
        print(f"[OK] Rows: {len(rows)}")

        missing = []
        for row in rows:
            check_image_paths(row, missing)

        if missing:
            print(f"[FAIL] Missing image paths: {len(missing)}")
            for item in missing[:10]:
                print("  ", item)
            has_failure = True
        else:
            print("[OK] Image path check passed")

        if rows:
            print("Sample fields:", list(rows[0].keys()))

    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
