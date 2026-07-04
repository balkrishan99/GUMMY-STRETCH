import urllib.request
import os

base_url = "https://gum-gum-hand-stretch-main.tiiny.site/"
files = [
    "gumgum.py",
    "arap.py",
    "requirements.txt",
    "README.md",
]

for file in files:
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(base_url + file, file)

# Download the models
os.makedirs("models", exist_ok=True)
models = [
    "face_landmarker.task",
    "hand_landmarker.task"
]
for model in models:
    if not os.path.exists(os.path.join("models", model)):
        print(f"Downloading {model}...")
        urllib.request.urlretrieve(base_url + model, os.path.join("models", model))

print("Done downloading files.")
