import sys
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faces_inference import FACESPredictor


def main():
    predictor = FACESPredictor(device="cpu")
    image = Image.new("RGB", (600, 800), "white")
    try:
        result = predictor.predict(image)
    except ValueError as exc:
        print(f"Face-detection check passed: {exc}")
        print("Model loading passed.")
        return

    etiology_sum = sum(item["probability"] for item in result["etiology"])
    subtype_sum = sum(item["probability"] for item in result["ss_subtypes"])
    assert abs(etiology_sum - 1.0) < 1e-4, etiology_sum
    assert abs(subtype_sum - 1.0) < 1e-4, subtype_sum
    print("Model loading and probability checks passed.")


if __name__ == "__main__":
    main()
