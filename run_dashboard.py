#!/usr/bin/env python3
"""
Smart Launcher for Hockey Analytics Dashboard
Automatically detects the best way to run the dashboard based on available environment.
"""

import sys
import os
import subprocess
import importlib.util

def check_streamlit_available():
    """Check if Streamlit is available"""
    return importlib.util.find_spec("streamlit") is not None

def check_matplotlib_backend():
    """Check if matplotlib can display plots"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        # Try to get the backend
        backend = matplotlib.get_backend()
        # Check if it's a GUI backend
        gui_backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'GTK4Agg', 'WXAgg', 'MacOSX']
        return backend in gui_backends
    except:
        return False

def main():
    print("Hockey Analytics Dashboard Launcher")
    print("=" * 40)
    
    # Check what's available
    streamlit_available = check_streamlit_available()
    matplotlib_gui_available = check_matplotlib_backend()
    
    print(f"Environment Check:")
    print(f"  Streamlit available: {'✅' if streamlit_available else '❌'}")
    print(f"  GUI plotting available: {'✅' if matplotlib_gui_available else '❌'}")
    print()
    
    # Determine best option
    if streamlit_available:
        print("Recommended: Streamlit Web App")
        print("   - Best user experience")
        print("   - Interactive interface")
        print("   - File upload capability")
        print()
        
        choice = input("Launch Streamlit app? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '']:
            print("Launching Streamlit app...")
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
            except KeyboardInterrupt:
                print("\nStreamlit app closed.")
            return
    
    # Fallback to standalone script
    print("Alternative: Standalone Python Script")
    print("   - Works in any Python environment")
    print("   - Command-line interface")
    print("   - Can save plots to files")
    print()
    
    choice = input("Launch standalone script? (y/n): ").lower().strip()
    if choice in ['y', 'yes', '']:
        print("Launching standalone script...")
        try:
            subprocess.run([sys.executable, "hockey_analytics.py"])
        except KeyboardInterrupt:
            print("\nStandalone script closed.")
        return
    
    print("No option selected. Exiting.")

if __name__ == "__main__":
    main() 