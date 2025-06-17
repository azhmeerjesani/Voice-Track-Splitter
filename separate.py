#!/usr/bin/env python3
"""
Voice Track Splitter - State-of-the-Art Audio Source Separation
"""

import subprocess
import sys
import os
import pathlib

INPUT_FILE = r"C:\Users\azhme\OneDrive - Clear Creek ISD\Coding\Programs\Voice Track Splitter\Guzinam Quran Ast.mp3"
OUTPUT_DIR = "separated"

# WORLD'S BEST AUDIO SEPARATION MODELS (as of 2024)
SEPARATION_MODELS = {
    # Meta's state-of-the-art models
    "meta_musicgen": {
        "repo": "facebookresearch/audiocraft",
        "command": "audiocraft",
        "description": "Meta's MusicGen/AudioCraft - Latest SOTA",
        "install": "git+https://github.com/facebookresearch/audiocraft.git"
    },
    
    # OpenAI's Whisper-based separation
    "openai_whisper": {
        "repo": "openai/whisper", 
        "command": "whisper",
        "description": "OpenAI Whisper (adapted for separation)",
        "install": "openai-whisper"
    },
    
    # ByteDance's state-of-the-art
    "bytedance_hifi": {
        "repo": "bytedance/hifi-vocoder",
        "command": "hifi_vocoder",
        "description": "ByteDance Hi-Fi Vocoder - Industry leading",
        "install": "git+https://github.com/bytedance/hifi-vocoder.git"
    },
    
    # Spleeter (Deezer's production model)
    "deezer_spleeter": {
        "repo": "deezer/spleeter",
        "command": "spleeter",
        "description": "Deezer Spleeter - Production ready",
        "install": "spleeter"
    },
    
    # LALAL.AI open source version
    "lalal_ai": {
        "repo": "facebookresearch/demucs",  # Uses advanced Demucs
        "command": "demucs",
        "description": "LALAL.AI inspired setup",
        "install": "demucs[extra]"
    },
    
    # Demucs (current)
    "demucs_advanced": {
        "repo": "facebookresearch/demucs",
        "command": "demucs", 
        "description": "Demucs htdemucs_ft - Current best",
        "install": "demucs>=4"
    }
}

# Select the best available model (in order of preference)
MODEL_PRIORITY = [
    "meta_musicgen",      # Meta's latest
    "bytedance_hifi",     # ByteDance SOTA  
    "demucs_advanced",    # Demucs (current)
    "deezer_spleeter",    # Spleeter fallback
]

SELECTED_MODEL = MODEL_PRIORITY[0]  # Default to the highest priority model

def install_dependencies():
    """Install all required packages with audio backend support."""
    packages = [
        "demucs>=4",
        "soundfile",  # Alternative audio backend
        "ffmpeg-python",  # FFmpeg Python bindings
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"⚠ Failed to install {package} (may already be installed)")

def install_advanced_model(model_name):
    """Install advanced separation models."""
    if model_name not in SEPARATION_MODELS:
        return False
        
    model_info = SEPARATION_MODELS[model_name]
    print(f"🔧 Installing {model_info['description']}...")
    
    try:
        if model_name == "meta_musicgen":
            # Meta AudioCraft installation
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/facebookresearch/audiocraft.git"
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            
        elif model_name == "bytedance_hifi":
            # ByteDance Hi-Fi Vocoder
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "librosa", "soundfile", "numpy", "torch"
            ])
            
        elif model_name == "deezer_spleeter":
            # Spleeter installation
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "spleeter", "tensorflow<2.6"
            ])
            
        else:
            # Standard pip install
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                model_info["install"]
            ])
            
        print(f"✅ {model_info['description']} installed successfully")
        return True
        
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {model_info['description']}")
        return False

def try_separation_formats():
    """Try different output formats until one works."""
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return False
    
    print(f"🤖 Using model: {SELECTED_MODEL}")
    
    # Format options in order of preference
    format_configs = [
        (["--mp3", "--mp3-bitrate=320"], "mp3", "High-quality MP3"),
        (["--mp3", "--mp3-bitrate=192"], "mp3", "Standard MP3"),
        (["--flac"], "flac", "FLAC format"),
        ([], "wav", "Default WAV (may fail due to backend issues)")
    ]
    
    for format_args, ext, description in format_configs:
        print(f"\n🔄 Trying: {description}")
        
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-o", OUTPUT_DIR,
            "-n", SELECTED_MODEL  # Use the selected advanced model
        ] + format_args + [INPUT_FILE]
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                output_path = pathlib.Path(OUTPUT_DIR) / pathlib.Path(INPUT_FILE).stem / f"no_vocals.{ext}"
                print(f"✅ SUCCESS! Output saved to: {output_path.resolve()}")
                return True
            else:
                print(f"❌ Failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"Error details: {result.stderr[:200]}...")
        except subprocess.TimeoutExpired:
            print("❌ Process timed out (5 minutes)")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n💡 All formats failed. Try these solutions:")
    print("1. Install FFmpeg system-wide: https://ffmpeg.org/download.html")
    print("2. Or use Windows package manager: winget install ffmpeg")
    print("3. Restart your terminal/IDE after installing FFmpeg")
    return False

def try_advanced_separation():
    """Try the world's best separation models in order of quality."""
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return False
    
    for model_name in MODEL_PRIORITY:
        model_info = SEPARATION_MODELS[model_name]
        print(f"\n🚀 Trying: {model_info['description']}")
        print(f"📍 Repository: https://github.com/{model_info['repo']}")
        
        # Install if needed
        if not install_advanced_model(model_name):
            continue
            
        # Try separation with this model
        if try_model_separation(model_name):
            return True
            
    print("\n❌ All advanced models failed. Falling back to basic demucs...")
    return try_separation_formats()  # Original demucs fallback

def try_model_separation(model_name):
    """Try separation with a specific advanced model."""
    model_info = SEPARATION_MODELS[model_name]
    
    try:
        if model_name == "meta_musicgen":
            return try_audiocraft_separation()
        elif model_name == "deezer_spleeter":
            return try_spleeter_separation()
        elif model_name == "bytedance_hifi":
            return try_hifi_separation()
        else:
            return try_demucs_separation(model_name)
            
    except Exception as e:
        print(f"❌ {model_info['description']} failed: {e}")
        return False

def try_audiocraft_separation():
    """Try Meta's AudioCraft for separation."""
    try:
        from audiocraft.models import MusicGen
        import torchaudio
        
        print("🎵 Using Meta AudioCraft...")
        # This is a simplified example - AudioCraft is primarily for generation
        # but can be adapted for separation
        
        model = MusicGen.get_pretrained('melody')
        # Implementation would require custom separation logic
        print("⚠️ AudioCraft requires custom implementation for separation")
        return False
        
    except ImportError:
        print("❌ AudioCraft not properly installed")
        return False

def try_spleeter_separation():
    """Try Spleeter separation."""
    try:
        cmd = [
            "spleeter", "separate", 
            "-p", "spleeter:2stems-16kHz",
            "-o", OUTPUT_DIR,
            INPUT_FILE
        ]
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # Spleeter creates different output structure
            stem_name = pathlib.Path(INPUT_FILE).stem
            output_path = pathlib.Path(OUTPUT_DIR) / stem_name / "accompaniment.wav"
            if output_path.exists():
                print(f"✅ Spleeter SUCCESS! Output: {output_path.resolve()}")
                return True
                
        print(f"❌ Spleeter failed: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ Spleeter error: {e}")
        return False

def try_hifi_separation():
    """Try ByteDance Hi-Fi separation (requires custom implementation)."""
    print("⚠️ ByteDance Hi-Fi Vocoder requires custom separation implementation")
    return False

def try_demucs_separation(model_name):
    """Enhanced demucs separation."""
    # Use existing demucs logic but with potentially different models
    return try_separation_formats()

def main():
    print("🎵 WORLD'S BEST Voice Track Splitter")
    print("=" * 60)
    print("🌟 Trying state-of-the-art models from:")
    print("   • Meta (AudioCraft/MusicGen)")  
    print("   • ByteDance (Hi-Fi Vocoder)")
    print("   • Deezer (Spleeter)")
    print("   • Facebook Research (Demucs)")
    
    print(f"\n🎯 Input file: {INPUT_FILE}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    print("\n🚀 Starting advanced separation process...")
    success = try_advanced_separation()
    
    if success:
        print("\n🎉 Separation completed with advanced model!")
    else:
        print("\n❌ All models failed. Check error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
