#!/usr/bin/env python3

import os
# Set environment variables for headless mode
os.environ['DISPLAY'] = ':99'  # Virtual display
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL instead of GLX

try:
    print("Importing Genesis...")
    import genesis as gs
    print("Genesis import successful!")
    
    # Try to initialize Genesis
    print("Initializing Genesis...")
    
    # Check what backends are available
    try:
        print("Available backends:")
        backends = gs.available_backends()
        print(backends)
    except Exception as e:
        print(f"Could not get available backends: {e}")
    
    # Try different backends
    backends_to_try = ['cuda', 'vulkan', 'cpu']
    
    for backend in backends_to_try:
        try:
            print(f"Trying backend: {backend}")
            gs.init(backend=backend)
            print(f"Successfully initialized with backend: {backend}")
            break
        except Exception as e:
            print(f"Failed with {backend}: {e}")
            continue
    else:
        print("Failed to initialize with any backend")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

