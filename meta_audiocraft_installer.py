#!/usr/bin/env python3
"""
Meta AudioCraft Installer and Test Script
"""

import subprocess
import sys
import os

def install_audiocraft():
    """Install Meta's AudioCraft with all dependencies."""
    print("🚀 Installing Meta AudioCraft - State-of-the-Art Audio AI")
    print("=" * 60)
    
    # Step 1: Install PyTorch with CUDA support
    print("\n📦 Step 1: Installing PyTorch with CUDA support...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        print("✅ PyTorch installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️ CUDA version failed, trying CPU version...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch", "torchaudio"
        ])
    
    # Step 2: Install AudioCraft from GitHub
    print("\n📦 Step 2: Installing AudioCraft from source...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/facebookresearch/audiocraft.git"
    ])
    print("✅ AudioCraft installed successfully")
    
    # Step 3: Install audio processing dependencies
    print("\n📦 Step 3: Installing audio processing libraries...")
    dependencies = [
        "demucs>=4",
        "soundfile", 
        "librosa", 
        "numpy", 
        "scipy",
        "ffmpeg-python"
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ])
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️ {dep} installation failed (may already be installed)")
    
    print("\n🎉 Meta AudioCraft installation completed!")
    
def test_audiocraft():
    """Test AudioCraft installation."""
    print("\n🧪 Testing AudioCraft installation...")
    
    try:
        # Test basic imports
        print("📊 Testing imports...")
        import torch
        print(f"✅ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}")
        
        import torchaudio
        print(f"✅ TorchAudio {torchaudio.__version__}")
        
        from audiocraft.models import MusicGen
        print("✅ AudioCraft MusicGen imported successfully")
        
        import demucs
        print(f"✅ Demucs available for separation")
        
        # Test model loading
        print("\n🤖 Testing model loading...")
        model = MusicGen.get_pretrained('melody')
        print("✅ MusicGen melody model loaded successfully")
        
        print("\n🎊 All tests passed! AudioCraft is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("Meta AudioCraft Setup Utility")
    print("=" * 40)
    
    choice = input("\nWhat would you like to do?\n1. Install AudioCraft\n2. Test installation\n3. Both\nEnter choice (1-3): ")
    
    if choice in ['1', '3']:
        install_audiocraft()
    
    if choice in ['2', '3']:
        test_audiocraft()
    
    print("\n🏁 Setup complete! You can now use Meta's AudioCraft for separation.")

if __name__ == "__main__":
    main()
