"""Baixa cs-training.csv via API do Kaggle. Exige ~/.kaggle/kaggle.json e aceite das regras da competição."""
import sys
import zipfile

from config import DATA_RAW, RAW_FILE, KAGGLE_COMPETITION


def main() -> int:
    if RAW_FILE.exists():
        print(f"Já existe: {RAW_FILE}")
        return 0
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_file(KAGGLE_COMPETITION, "cs-training.csv", path=str(DATA_RAW))
        zipped = DATA_RAW / "cs-training.csv.zip"
        if zipped.exists():
            with zipfile.ZipFile(zipped) as z:
                z.extract("cs-training.csv", DATA_RAW)
            zipped.unlink()
    except Exception as e:  # token ausente, regras não aceitas, sem rede...
        print(f"Download automático indisponível ({type(e).__name__}).")
        print("Baixe manualmente em https://www.kaggle.com/c/GiveMeSomeCredit/data")
        print(f"(aceite as regras da competição) e coloque cs-training.csv em: {DATA_RAW}")
        return 1
    print(f"OK: {RAW_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
