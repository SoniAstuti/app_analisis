import nltk

print("Mulai mengunduh resource NLTK...")
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    print("\n[OK] Semua resource berhasil diunduh!")
except Exception as e:
    print(f"\n[ERROR] Terjadi masalah saat mengunduh: {e}")