import requests
import json
import os

url = 'http://localhost:8000/api/segment'
image_path = r'd:\MedFusion\MedFusion\data\Kidney_Stone\train\images\1-3-46-670589-33-1-63700700749865510700001-5062181202000819812_png_jpg.rf.269520bcaab75e008e00f57f3fa98851.jpg'

if not os.path.exists(image_path):
    print("Image not found:", image_path)
    exit(1)

files = {'image': open(image_path, 'rb')}
# Test with a bounding box prompt
data = {'box': json.dumps([50, 50, 200, 200])}

print("Sending request to server...")
try:
    response = requests.post(url, files=files, data=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
