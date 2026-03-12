import requests
import json
import os

url = 'http://localhost:8000/api/segment'
health_url = 'http://localhost:8000/health'

# Find the first image in Kidney_Stone/train/images
data_dir = os.path.join(os.path.dirname(__file__), 'data', 'Kidney_Stone', 'train', 'images')
images = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')] if os.path.exists(data_dir) else []

if not images:
    print("No images found in the dataset to test with. Looking at:", data_dir)
    exit(1)

image_path = os.path.join(data_dir, images[0])
print(f"Testing with image: {image_path}")

files = {'image': open(image_path, 'rb')}
# Test without a bounding box prompt to verify auto-GT label detection
data = {}

print("Checking health...")
try:
    response = requests.get(health_url)
    print("Health Status:", response.status_code)
    print("Health Response:", response.json())
except Exception as e:
    print("Health Check Error:", e)

print("Sending request to server...")
try:
    response = requests.post(url, files=files, data=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
