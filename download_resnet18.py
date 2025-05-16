import os
import requests

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url)
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {dest_path}")

# 첫 번째 파일 다운로드
download_file(
    url="https://pjreddie.com/media/files/resnet18.weights",
    dest_path="weights/resnet18.weights"
)

# 두 번째 파일 다운로드
download_file(
    url="https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/resnet18.cfg",
    dest_path="cfg/resnet18.cfg"
)
