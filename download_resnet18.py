import os
import requests

def download_file_if_not_exists(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()  # 요청 실패 시 예외 발생
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {dest_path}")

# 첫 번째 파일 다운로드 (없을 때만)
download_file_if_not_exists(
    url="https://pjreddie.com/media/files/resnet18.weights",
    dest_path="weights/resnet18.weights"
)

# 두 번째 파일 다운로드 (없을 때만)
download_file_if_not_exists(
    url="https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/resnet18.cfg",
    dest_path="cfg/resnet18.cfg"
)
