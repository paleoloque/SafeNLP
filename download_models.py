""" 
download_models.py 

A short helper for downloading models from GitHub Release. 
"""

import os, urllib.request

RELEASE_TAG = "v1.0.0"
BASE = f"https://github.com/paleoloque/SafeNLP/releases/download/{RELEASE_TAG}"
FILES = {
    "lgbm_model.joblib": f"{BASE}/lgbm_model.joblib",
    "w2v.kv":            f"{BASE}/w2v.model",
    "threshold.txt":     f"{BASE}/threshold.txt",
}

os.makedirs("artifacts", exist_ok=True)
for name, url in FILES.items():
    dst = os.path.join("artifacts", name)
    print("->", dst)
    urllib.request.urlretrieve(url, dst)

print("Done.")
