#!/usr/bin/env python3
"""
Test script to verify EgoBlur installation with UV
"""

import sys
import subprocess
import importlib

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    packages = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'moviepy'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ OpenCV version: {cv2.__version__}")
            elif package == 'torch':
                import torch
                print(f"✓ PyTorch version: {torch.__version__}")
                print(f"✓ CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"✓ CUDA version: {torch.version.cuda}")
            elif package == 'torchvision':
                import torchvision
                print(f"✓ TorchVision version: {torchvision.__version__}")
            elif package == 'numpy':
                import numpy
                print(f"✓ NumPy version: {numpy.__version__}")
            elif package == 'moviepy':
                import moviepy
                print(f"✓ MoviePy version: {moviepy.__version__}")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_demo_script():
    """Test that the demo script can be run"""
    print("\nTesting demo script...")
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, 'script/demo_ego_blur.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Demo script help command works!")
            return True
        else:
            print(f"❌ Demo script failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Demo script timed out")
        return False
    except Exception as e:
        print(f"❌ Demo script error: {e}")
        return False

def main():
    """Main test function"""
    print("EgoBlur UV Installation Test")
    print("=" * 30)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test demo script
    script_ok = test_demo_script()
    
    # Summary
    print("\n" + "=" * 30)
    if imports_ok and script_ok:
        print("🎉 All tests passed! EgoBlur is ready to use.")
        print("\nNext steps:")
        print("1. Download models from: https://www.projectaria.com/tools/egoblur")
        print("2. Run: uv run python script/demo_ego_blur.py --help")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
