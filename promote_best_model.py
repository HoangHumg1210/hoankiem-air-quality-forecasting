from __future__ import annotations

import argparse
from pathlib import Path

from bundle_registry import promote_best_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chọn bundle tốt nhất từ model_registry và di chuyển vào best_model_bundle.",
    )
    parser.add_argument(
        "--bundle-key",
        help="Chọn một bundle cụ thể thay vì chọn bundle có MAE thấp nhất.",
    )
    parser.add_argument(
        "--app-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Dự án chứa model_registry và best_model_bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()  # Đọc tham số truyền vào từ dòng lệnh

    result = promote_best_bundle(args.app_dir, bundle_key=args.bundle_key)
    # Chọn và đưa bundle tốt nhất lên làm bundle chính thức

    print(f"Promoted model: {result['model_name']}")      # Tên model được chọn
    print(f"Bundle key: {result['bundle_key']}")          # Mã bundle
    print(f"MAE: {result['mae']:.4f}")                    # Sai số MAE
    print(f"Source: {result['source_bundle_dir']}")       # Thư mục nguồn
    print(f"Target: {result['target_bundle_dir']}")       # Thư mục đích


if __name__ == "__main__":
    main()
