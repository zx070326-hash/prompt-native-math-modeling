from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from common import PROJECT_ROOT, sha256


def run():
    stage = (PROJECT_ROOT / "tmp" / "support_package").resolve()
    expected_parent = (PROJECT_ROOT / "tmp").resolve()
    if stage.parent != expected_parent:
        raise RuntimeError("Unsafe staging target")
    if stage.exists():
        raise RuntimeError(f"Staging path already exists; refusing destructive overwrite: {stage}")
    build_archive = (expected_parent / "000_支撑材料.build.zip").resolve()
    if build_archive.parent != expected_parent:
        raise RuntimeError("Unsafe temporary archive target")
    if build_archive.exists():
        raise RuntimeError(f"Temporary archive already exists; refusing destructive overwrite: {build_archive}")
    output = (PROJECT_ROOT / "outputs" / "000_支撑材料.zip").resolve()
    if output.parent != (PROJECT_ROOT / "outputs").resolve():
        raise RuntimeError("Unsafe support archive target")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing support archive: {output}")
    stage.mkdir(parents=True)

    def copy_file(relative: str, target_relative: str | None = None):
        src = PROJECT_ROOT / relative
        dst = stage / (target_relative or relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for name in ["README.md", "requirements.txt"]:
        copy_file(name)
    for folder, patterns in {
        "src": ["*.py"], "outputs": ["*.csv"], "figures": ["*.png"],
        "data/external": ["*.json"], "paper": ["*.tex"], "support": ["*.tex"],
    }.items():
        for pattern in patterns:
            for src in sorted((PROJECT_ROOT / folder).glob(pattern)):
                copy_file(str(src.relative_to(PROJECT_ROOT)))
    for src in sorted((PROJECT_ROOT / "logs").glob("*.json")):
        if src.name == "support_package.json":
            continue
        copy_file(str(src.relative_to(PROJECT_ROOT)))
    copy_file("outputs/pdf/AI 工具使用详情.pdf", "AI 工具使用详情.pdf")

    manifest = {
        str(path.relative_to(stage)).replace("\\", "/"): sha256(path)
        for path in sorted(stage.rglob("*")) if path.is_file()
    }
    manifest_path = stage / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with zipfile.ZipFile(build_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, str(path.relative_to(stage)).replace("\\", "/"))
    expected_members = sorted([*manifest, "MANIFEST.json"])
    with zipfile.ZipFile(build_archive) as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Support archive CRC failure: {bad_member}")
        if sorted(zf.namelist()) != expected_members:
            raise RuntimeError("Support archive members do not match the generated manifest")
    if build_archive.stat().st_size >= 20 * 1024 * 1024:
        raise RuntimeError(f"Support archive exceeds 20 MiB: {build_archive.stat().st_size}")
    build_archive.replace(output)
    report = {
        "archive": output.relative_to(PROJECT_ROOT).as_posix(), "bytes": output.stat().st_size,
        "sha256": sha256(output), "file_count": len(manifest) + 1,
        "under_20MiB": output.stat().st_size < 20 * 1024 * 1024,
    }
    (PROJECT_ROOT / "logs" / "support_package.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
