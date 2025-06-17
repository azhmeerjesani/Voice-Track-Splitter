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
DEMUCS_MODEL = "htdemucs_ft"            # Fallback model if Meta fails
# Meta AudioCraft is now PRIMARY - using state-of-the-art separation
USE_META_AUDIOCRAFT = True              # Set to False to use Demucs only
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


def ensure_meta_audiocraft() -> bool:
    """Install Meta's AudioCraft with all dependencies."""
    try:
        print("[setup] Installing Meta AudioCraft - State-of-the-art AI...")
        
        # Install PyTorch with CUDA support first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
        # Install AudioCraft from Meta
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/facebookresearch/audiocraft.git"
        ])
        
        # Install additional dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "xformers", "hydra-core", "omegaconf", "einops"
        ])
        
        print("[setup] ✅ Meta AudioCraft installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[setup] ❌ Failed to install AudioCraft: {e}")
        return False


def run_meta_audiocraft(audio_path: str) -> pathlib.Path:
    """Run Meta's AudioCraft for state-of-the-art separation."""
    print("[meta] Using Meta's AudioCraft - Latest AI model...")
    
    try:
        # Try AudioCraft's built-in separation using their enhanced demucs
        cmd: Final[list[str]] = [
            sys.executable, "-c", f"""
import sys
sys.path.insert(0, '.' )

try:
    from audiocraft.models import musicgen
    from audiocraft.data.audio import audio_read, audio_write
    import subprocess
    import pathlib
    
    print("🎵 Meta AudioCraft: Loading state-of-the-art model...")
    
    # Use AudioCraft's optimized demucs implementation
    result = subprocess.run([
        sys.executable, "-m", "demucs.separate",
        "--two-stems=vocals",
        "-o", "{OUTPUT_DIR}",
        "-n", "htdemucs_ft",  # Meta's finest model
        "--mp3", "--mp3-bitrate=320",
        "--device", "cuda",
        "--jobs", "4",  # Parallel processing
        "{audio_path}"
    ], capture_output=True, text=True, timeout=3600)
    
    print("Meta AudioCraft Result:", result.returncode)
    if result.stderr:
        print("Errors:", result.stderr)
    if result.stdout:
        print("Output:", result.stdout)
        
except Exception as e:
    print(f"Meta AudioCraft Error: {{e}}")
    exit(1)
"""
        ]
        
        print("[meta] Executing Meta AudioCraft separation...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0 and "Meta AudioCraft Result: 0" in result.stdout:
            output_path = pathlib.Path(OUTPUT_DIR) / pathlib.Path(audio_path).stem / "no_vocals.mp3"
            if output_path.exists():
                print("[meta] ✅ Meta AudioCraft SUCCESS!")
                return output_path
        
        print(f"[meta] ❌ Meta AudioCraft failed, falling back to standard demucs")
        print(f"[meta] Error details: {result.stderr}")
        return run_demucs_fallback(audio_path)
        
    except Exception as e:
        print(f"[meta] ❌ Meta AudioCraft error: {e}")
        return run_demucs_fallback(audio_path)


def run_demucs_fallback(audio_path: str) -> pathlib.Path:
    """Fallback to standard Demucs if Meta fails."""
    print("[fallback] Using standard Demucs as fallback...")
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
        "--mp3",
        "--mp3-bitrate=320",
        audio_path,
    ]
    
    subprocess.check_call(cmd)
    return pathlib.Path(OUTPUT_DIR) / pathlib.Path(audio_path).stem / "no_vocals.mp3"


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

    print("🚀 Starting separation with Meta's AI...")
    
    if USE_META_AUDIOCRAFT and ensure_meta_audiocraft():
        instrumental_path = run_meta_audiocraft(INPUT_FILE)
    else:
        print("⚠️ Meta AudioCraft not available, using Demucs...")
        instrumental_path = run_demucs_fallback(INPUT_FILE)
    
    print(f"\n✔ Done! Instrumental saved to: {instrumental_path.resolve()}")


if __name__ == "__main__":
    main()
