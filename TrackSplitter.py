#!/usr/bin/env python3
"""
extract_instrumental_with_demucs.py

One‑click utility: double‑click or run `python extract_instrumental_with_demucs.py`
and it delivers an **instrumental MP3**—no command‑line flags required.

Steps to use
------------
1. Edit **INPUT_FILE** below (and optionally OUTPUT_DIR / DEMUCS_MODEL).
2. Run the script. It will install Demucs v4+ on first run if necessary.

What happens under the hood
---------------------------
* Installs/updates Demucs in your current Python environment.
* Calls `demucs --two-stems=vocals` to remove vocals.
* Places the instrumental at
    OUTPUT_DIR/<song‑name>/no_vocals.mp3

"""

# ========= USER‑CONFIGURABLE VARIABLES =========
INPUT_FILE   = r"C:\Users\azhme\OneDrive - Clear Creek ISD\Coding\Programs\Voice Track Splitter\Guzinam Quran Ast.mp3"  # ← Put your MP3/WAV/FLAC here
OUTPUT_DIR   = "separated"              # Where the results will be stored
DEMUCS_MODEL = "htdemucs_ft"            # Most advanced model (fine-tuned hybrid transformer)
# Alternative models (in order of quality vs speed):
# "htdemucs_ft"    - Best quality, slowest (fine-tuned hybrid transformer) ← RECOMMENDED
# "htdemucs_6s"    - 6-stem separation (vocals, drums, bass, piano, guitar, other)
# "htdemucs"       - Good quality, faster (hybrid transformer)
# "hdemucs_mmi"    - Older but still good
# "mdx_extra"      - Alternative architecture
# ==============================================

import importlib.util
import os
import pathlib
import subprocess
import sys
from typing import Final


def ensure_demucs() -> None:
    """Install Demucs and audio backends via pip if not available."""
    packages_to_install = []
    
    if importlib.util.find_spec("demucs") is None:
        packages_to_install.append("demucs>=4")
    
    # Install audio backend dependencies to avoid torchaudio errors
    if importlib.util.find_spec("soundfile") is None:
        packages_to_install.append("soundfile")
    
    try:
        import ffmpeg
    except ImportError:
        packages_to_install.append("ffmpeg-python")
    
    if packages_to_install:
        print(f"[setup] Installing: {', '.join(packages_to_install)}")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
        ] + packages_to_install)


def run_demucs(audio_path: str) -> pathlib.Path:
    """Run Demucs and return the path to the instrumental MP3."""
    ensure_demucs()

    cmd: Final[list[str]] = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-o",
        OUTPUT_DIR,
        "-n",
        DEMUCS_MODEL,
        "--mp3",  # Force MP3 output to avoid torchaudio backend issues
        "--mp3-bitrate=320",  # High quality
        audio_path,
    ]
    print("[demucs] Executing:", " ".join(cmd))
    
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[error] Demucs failed with exit code {e.returncode}")
        print("[info] If you see audio backend errors, try installing ffmpeg:")
        print("       Download from: https://ffmpeg.org/download.html")
        print("       Or run: winget install ffmpeg")
        raise

    return pathlib.Path(OUTPUT_DIR) / pathlib.Path(audio_path).stem / "no_vocals.mp3"


def main() -> None:
    if INPUT_FILE == "path/to/your/song.mp3":
        sys.exit("✘ Please set INPUT_FILE at the top of the script before running.")

    if not os.path.isfile(INPUT_FILE):
        sys.exit(f"✘ Input file not found: {INPUT_FILE}")

    instrumental_path = run_demucs(INPUT_FILE)
    print(f"\n✔ Done! Instrumental saved to: {instrumental_path.resolve()}")


if __name__ == "__main__":
    main()
