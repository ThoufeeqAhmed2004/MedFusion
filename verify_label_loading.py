import os
import sys
import cv2
import numpy as np
import subprocess
import shutil

def create_dummy_data():
    # Create a 100x100 black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite("test_image.jpg", img)
    
    # Create a label file: class 0, center 0.5, 0.5, size 0.2, 0.2
    # box should be 40, 40, 60, 60
    with open("test_image.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

def run_inference(image_path):
    cmd = [
        sys.executable, "src/main.py",
        "--image_path", image_path,
        "--device", "cpu", # Use CPU for test
        "--dataset", "Kidney_Stone" # Dummy dataset name
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr

def run_test():
    create_dummy_data()
    
    print("--- Test 1: Image with Label ---")
    output = run_inference("test_image.jpg")
    if "Found label file" in output and "Loaded 1 boxes" in output:
        print("[PASS] Label detected and loaded.")
    else:
        print("[FAIL] Label not detected or loaded correctly.")
        print(output)

    print("\n--- Test 2: Image without Label ---")
    os.rename("test_image.txt", "test_image.txt.bak")
    output = run_inference("test_image.jpg")
    if "No label file found" in output:
        print("[PASS] Correctly identified no label file.")
    else:
        print("[FAIL] Should have reported no label file.")
        print(output)
        
    # Cleanup
    if os.path.exists("test_image.jpg"): os.remove("test_image.jpg")
    if os.path.exists("test_image.txt"): os.remove("test_image.txt")
    if os.path.exists("test_image.txt.bak"): os.remove("test_image.txt.bak")

if __name__ == "__main__":
    run_test()
