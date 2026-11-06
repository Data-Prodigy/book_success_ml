"""
Tidy the project folder: keep ONLY what the current project needs, move everything else to _cleanup_trash/.

    python cleanup_project.py            # dry run: prints what would move, changes nothing
    python cleanup_project.py --apply    # moves it (nothing is deleted; you delete _cleanup_trash/ yourself)

Run it from the project root (the folder that holds docker-compose.yaml). Moving, not deleting, means a
mistake costs you one drag-and-drop instead of a file.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
TRASH = ROOT / "_cleanup_trash"

# ---- what the project needs -------------------------------------------------------------------------------
# Top level: name -> True (keep everything inside) | set of allowed names inside (anything else is moved)
KEEP_ALL = True
TOP = {
    ".env": KEEP_ALL, ".gitignore": KEEP_ALL, ".dockerignore": KEEP_ALL, ".git": KEEP_ALL, ".github": KEEP_ALL,
    ".vscode": KEEP_ALL, "venv": KEEP_ALL,                      # your editor settings + local virtual environment
    "docker-compose.yaml": KEEP_ALL, "Dockerfile.hops": KEEP_ALL, "pytest.ini": KEEP_ALL, "README.md": KEEP_ALL,
    "requirements.txt": KEEP_ALL, "requirements_hops.txt": KEEP_ALL, "requirements-dev.txt": KEEP_ALL,
    "state": KEEP_ALL, "mlflow_data": KEEP_ALL, "inference_metrics": KEEP_ALL,   # written by docker compose / the pipelines
    "raw_openlibrary_v2": KEEP_ALL,                                                # fetch cache: re-fetching is free
    "data": {"openlibrary_books.csv"},
    "api": {"fast_api.py", "__init__.py"},
    "test": {"conftest.py", "test_core.py"},
    "src": {"config.py", "__init__.py", "features"},
}
FEATURES = {"author_history.py", "evaluation.py", "feature_pipeline.py", "fetch_openlibrary.py", "genre.py",
            "hopsworks_utils.py", "inference_pipeline.py", "labeling.py", "model_registry.py", "train_pipeline.py"}
NEVER_TOUCH_INSIDE = {"venv", ".git"}


def plan():
    moves = []
    for item in sorted(ROOT.iterdir()):
        name = item.name
        if name == TRASH.name or name == Path(__file__).name:
            continue
        rule = TOP.get(name)
        if rule is None:
            moves.append(item)
        elif rule is not KEEP_ALL:
            for child in sorted(item.iterdir()):
                allowed = child.name in rule
                if allowed and item.name == "src" and child.name == "features":
                    for f in sorted(child.iterdir()):
                        if f.name not in FEATURES:
                            moves.append(f)
                elif not allowed:
                    moves.append(child)
    # __pycache__ folders inside the kept code folders
    for keep in ("api", "src", "test"):
        for p in (ROOT / keep).rglob("__pycache__") if (ROOT / keep).exists() else []:
            if p not in moves and not any(m in p.parents for m in moves):
                moves.append(p)
    return moves


def main():
    apply = "--apply" in sys.argv
    if not (ROOT / "src" / "features" / "feature_pipeline.py").exists() and not (ROOT / "docker-compose.yaml").exists():
        sys.exit("Run this from the project root (the folder with docker-compose.yaml). Nothing changed.")

    # files that belong in src/features but sit elsewhere / are spelled differently
    fixes = []
    if (ROOT / "fetch_openlibrary.py").exists() and not (ROOT / "src/features/fetch_openlibrary.py").exists():
        fixes.append((ROOT / "fetch_openlibrary.py", ROOT / "src/features/fetch_openlibrary.py"))
    old, new = ROOT / "src/features/hopswork_utils.py", ROOT / "src/features/hopsworks_utils.py"
    if old.exists() and not new.exists():
        fixes.append((old, new))

    moves = [m for m in plan() if m not in {a for a, _ in fixes}]
    print(f"{'MOVING' if apply else 'WOULD MOVE'} {len(moves)} item(s) to {TRASH.name}/ :")
    for m in moves:
        print("   ", m.relative_to(ROOT), "/" if m.is_dir() else "")
    for a, b in fixes:
        print(f"{'RENAMING' if apply else 'WOULD RENAME'}: {a.relative_to(ROOT)} -> {b.relative_to(ROOT)}")
    if not apply:
        print("\nDry run only. Re-run with --apply to do it.")
        return
    for a, b in fixes:
        shutil.move(str(a), str(b))
    for m in moves:
        dest = TRASH / m.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(m), str(dest))
    print(f"\nDone. Check that everything still works, then delete the {TRASH.name} folder.")


if __name__ == "__main__":
    main()