import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
OUTPUT = Path("Online_Retail.xlsx")

if OUTPUT.exists():
    print(f"{OUTPUT} sudah tersedia.")
else:
    print("Mengunduh Online Retail dari UCI...")
    urllib.request.urlretrieve(URL, OUTPUT)
    print("Download selesai:", OUTPUT)
