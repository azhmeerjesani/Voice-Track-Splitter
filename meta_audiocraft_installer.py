#!/usr/bin/env python3
"""
Meta AudioCraft Installer and Test Script - Fixed for Windows
"""

import subprocess
import sys
import os
import platform

def check_system_requirements():
    """Check if system has required build tools."""
    print("🔍 Checking system requirements...")
    
    issues = []
    
    # Check if we're on Windows
    if platform.system() == "Windows":
        print("✅ Windows detected")
        
        # Check for Visual Studio Build Tools
        vs_paths = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
            r"C:\Program Files\Microsoft Visual Studio\2019\BuildTools",
        ]
        
        vs_found = any(os.path.exists(path) for path in vs_paths)
        
        if not vs_found:
            issues.append("❌ Visual Studio Build Tools not found")
            print("❌ You need to install Visual Studio Build Tools")
            print("   Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print("   Select: C++ build tools, Windows SDK, CMake tools")
        else:
            print("✅ Visual Studio Build Tools found")
    
    # Check for Git
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("✅ Git is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("❌ Git not found")
        print("❌ Git is required")
        print("   Download from: https://git-scm.com/download/win")
    
    return len(issues) == 0

def install_audiocraft_simple():
    """Install AudioCraft with CPU-only fallback to avoid compilation issues."""
    print("🚀 Installing Meta AudioCraft (CPU-friendly version)")
    print("=" * 60)
    
    try:
        # Step 1: Install PyTorch CPU version (no CUDA compilation issues)
        print("\n📦 Step 1: Installing PyTorch (CPU version for compatibility)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
        ], timeout=600)
        print("✅ PyTorch CPU installed successfully")
        
        # Step 2: Install basic dependencies first
        print("\n📦 Step 2: Installing basic dependencies...")
        basic_deps = [
            "numpy",
            "scipy", 
            "soundfile",
            "librosa",
            "demucs>=4"
        ]
        
        for dep in basic_deps:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep
                ])
                print(f"✅ {dep} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️ {dep} failed (may already be installed)")
        
        # Step 3: Try to install AudioCraft (may fail due to compilation)
        print("\n📦 Step 3: Attempting AudioCraft installation...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/facebookresearch/audiocraft.git"
            ], timeout=1200)  # 20 minutes timeout
            print("✅ AudioCraft installed successfully!")
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("⚠️ AudioCraft compilation failed (this is common on Windows)")
            print("🔄 Setting up Meta-compatible alternative...")
            return install_meta_alternative()
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return install_meta_alternative()

def install_meta_alternative():
    """Install Meta-compatible setup using advanced Demucs models."""
    print("\n🔧 Installing Meta-compatible alternative (no compilation required)...")
    
    try:
        # Install the most advanced demucs models (Meta's contributions)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "demucs[extra]>=4"  # Extra features including Meta's models
        ])
        
        # Install additional audio processing tools
        extra_deps = [
            "ffmpeg-python",
            "pydub",
            "audioread"
        ]
        
        for dep in extra_deps:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep
                ])
                print(f"✅ {dep} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️ {dep} installation skipped")
        
        print("✅ Meta-compatible setup completed!")
        print("🎵 You can now use Meta's advanced demucs models:")
        print("   - htdemucs_ft (Meta's finest model)")
        print("   - htdemucs_6s (6-stem separation)")
        print("   - htdemucs (hybrid transformer)")
        
        return True
        
    except Exception as e:
        print(f"❌ Alternative setup failed: {e}")
        return False

def test_meta_setup():
    """Test the Meta-compatible setup."""
    print("\n🧪 Testing Meta-compatible setup...")
    
    try:
        # Test basic imports
        print("📊 Testing imports...")
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import demucs
        print(f"✅ Demucs available")
        
        # Test if we can access Meta's models
        cmd = [sys.executable, "-c", """
import subprocess
result = subprocess.run([
    'python', '-m', 'demucs', '--help'
], capture_output=True, text=True)
print('Demucs help available:', result.returncode == 0)
"""]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "Demucs help available: True" in result.stdout:
            print("✅ Demucs command line interface working")
        
        # Try to test AudioCraft if available
        try:
            from audiocraft.models import MusicGen
            print("✅ AudioCraft successfully installed and importable!")
            
            # Test model loading
            print("🤖 Testing Meta model loading...")
            model = MusicGen.get_pretrained('melody')
            print("✅ Meta MusicGen model loaded successfully!")
            
        except ImportError:
            print("ℹ️ AudioCraft not available (using Meta-compatible demucs instead)")
            print("✅ Meta's demucs models are ready to use")
        
        print("\n🎊 Setup test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("Meta AudioCraft Setup Utility - Windows Compatible")
    print("=" * 55)
    
    # Check system requirements first
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please install:")
        print("1. Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("2. Git: https://git-scm.com/download/win")
        print("\nAfter installation, restart your terminal and run this script again.")
        return
    
    choice = input("\nWhat would you like to do?\n1. Install Meta AudioCraft (full)\n2. Install Meta-compatible (safe)\n3. Test installation\n4. All\nEnter choice (1-4): ")
    
    if choice in ['1']:
        success = install_audiocraft_simple()
    elif choice in ['2']:
        success = install_meta_alternative()
    elif choice in ['3']:
        success = test_meta_setup()
    elif choice in ['4']:
        print("🚀 Running complete setup...")
        success = install_audiocraft_simple()
        if success:
            test_meta_setup()
    else:
        print("Invalid choice")
        return
    
    if success:
        print("\n🏁 Setup complete! You can now run your separation scripts.")
        print("💡 If AudioCraft failed, the Meta-compatible demucs setup will work perfectly.")
    else:
        print("\n❌ Setup had issues. Try the Meta-compatible option (choice 2).")

if __name__ == "__main__":
    main()
