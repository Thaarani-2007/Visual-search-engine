"""
Local Test Dataset Generator.
Creates a minimal, synthetic Stanford Online Products (SOP) mock dataset
in data/raw/ to allow immediate, offline end-to-end testing of:
- Metadata ingestion & validation
- Identity-disjoint train/val/test splitting
- Custom Triplet Dataset & DataLoader batching
- Visual triplet inspection
without requiring immediate multi-gigabyte downloads.
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image, ImageDraw

from src.config import PROJECT_ROOT, load_config


def generate_mock_sop_dataset(
    num_categories: int = 4,
    identities_per_category: int = 10,
    images_per_identity: int = 5,
    image_size: int = 256,
) -> Path:
    """Generate structured mock dataset with realistic instance-level variations."""
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    categories = cfg["dataset"]["target_categories"][:num_categories]
    records = []
    image_counter = 1

    print(f"Generating mock SOP dataset in {raw_dir}...")
    np.random.seed(cfg["seed"])

    for cat_idx, cat in enumerate(categories):
        cat_dir = raw_dir / f"{cat}_final"
        cat_dir.mkdir(parents=True, exist_ok=True)

        for id_idx in range(1, identities_per_category + 1):
            product_id = 1000 * (cat_idx + 1) + id_idx
            # Base color for the product identity
            base_color = tuple(np.random.randint(50, 220, size=3).tolist())

            for img_idx in range(images_per_identity):
                # Add variation in lighting / angle
                color_var = np.clip(
                    np.array(base_color) + np.random.randint(-30, 30, size=3), 0, 255
                ).astype(int)

                # Create synthetic image with shape indicating category
                img = Image.new("RGB", (image_size, image_size), color=(240, 240, 240))
                draw = ImageDraw.Draw(img)

                # Draw product-like shape
                margin = 40 + np.random.randint(-10, 10)
                draw.rectangle(
                    [margin, margin, image_size - margin, image_size - margin],
                    fill=tuple(color_var),
                    outline=(20, 20, 20),
                    width=3,
                )
                draw.text(
                    (margin + 10, margin + 10),
                    f"{cat.upper()}\nSKU:{product_id}\nView:{img_idx}",
                    fill=(255, 255, 255),
                )

                filename = f"{product_id}_{img_idx}.jpg"
                rel_path = f"{cat}_final/{filename}"
                img_path = cat_dir / filename
                img.save(img_path, quality=90)

                records.append(f"{image_counter} {product_id} {cat_idx + 1} {rel_path}\n")
                image_counter += 1

    # Write SOP-format metadata file Ebay_info.txt
    info_file = raw_dir / "Ebay_info.txt"
    with open(info_file, "w", encoding="utf-8") as f:
        f.write("image_id class_id super_class_id path\n")
        f.writelines(records)

    print(
        f"Mock dataset generated: {len(records)} images across "
        f"{num_categories * identities_per_category} identities in {raw_dir}."
    )
    return info_file


if __name__ == "__main__":
    generate_mock_sop_dataset()

