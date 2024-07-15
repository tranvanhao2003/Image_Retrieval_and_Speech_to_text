import requests

url = "http://0.0.0.0:8001/process_voice/"
file_path = "/home/ai_dev/work_space/production/Speech_to_text/test_voice/test.wav"

with open(file_path, "rb") as f:
    files = {"data_type": f}
    response = requests.post(url, files=files)

print(response.json())
