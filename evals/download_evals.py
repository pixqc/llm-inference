import os
import shutil
import tarfile

import requests


def download_and_extract(url, tar_path, extract_path, final_path):
  if not os.path.exists(tar_path):
    print(f"Downloading {url}...")
    response = requests.get(url)
    with open(tar_path, "wb") as f:
      f.write(response.content)
  else:
    print(f"Using cached download at {tar_path}")

  with tarfile.open(tar_path) as tar:
    tar.extractall(extract_path)
  nested_dir = os.path.join(extract_path, os.listdir(extract_path)[0])
  for item in os.listdir(nested_dir):
    src = os.path.join(nested_dir, item)
    dst = os.path.join(final_path, item)
    if os.path.exists(dst):
      shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
    shutil.move(src, dst)
  shutil.rmtree(extract_path)


if __name__ == "__main__":
  datasets = [
    {
      "url": "https://people.eecs.berkeley.edu/~hendrycks/data.tar",
      "tar": "/tmp/data.tar",
      "final": "evals/mmlu",
    },
    {
      "url": "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar",
      "tar": "/tmp/MATH.tar",
      "final": "evals/math",
    },
  ]
  for data in datasets:
    os.makedirs(data["final"], exist_ok=True)
    download_and_extract(data["url"], data["tar"], "/tmp/temp_extract", data["final"])

  print("Dataset download complete!")
