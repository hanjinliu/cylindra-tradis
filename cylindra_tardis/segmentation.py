from __future__ import annotations
import subprocess
from dataclasses import dataclass
from pathlib import Path
import impy as ip

@dataclass
class TardisSegmentationModel:
    prefilter_cutoff_nm: float
    min_length: int = 50
    patch_size: int = 128
    cnn_threshold: float = 0.25
    dist_threshold: float = 0.5
    device: str = "0"
    connect_splines: int = 250
    connect_cylinder: int = 25

    def tardis_dir(self, image_path: Path) -> Path:
        return image_path.parent / "TARDIS"

    def run(self, img: ip.ImgArray | ip.LazyImgArray, image_path: Path):
        overlap = [min(s, 32) for s in img.shape]
        _tiled = img.tiled(chunks=(224, 224, 224), overlap=overlap)
        sigma = self.prefilter_cutoff_nm / img.scale.x
        tardis_dir = self.tardis_dir(image_path)
        img_filt = _tiled.lowpass_filter(cutoff=1.0 / sigma)
        img_filt.imsave(image_path.parent / "input.mrc")
        
        tardis_dir = self.tardis_dir(image_path)
        tardis_dir.mkdir(exist_ok=True)
        basename = image_path.stem

        # Run prediction
        args = [
            "tardis_mt",
            "-dir", str(image_path.parent / "input.mrc"),
            "-out", "mrc_csv",
            "--filter_by_length", str(self.min_length * 10),
            "--patch_size", str(self.patch_size),
            "--cnn_threshold", str(round(self.cnn_threshold, 3)),
            "--dist_threshold", str(round(self.dist_threshold, 3)),
            "--device", self.device,
            "--connect_splines", str(self.connect_splines * 10),
            "--connect_cylinder", str(self.connect_cylinder * 10),
        ]  # fmt: skip
        subprocess.run(args, check=True, cwd=str(tardis_dir))

        prediction_dir = tardis_dir / "Predictions"
        for file in prediction_dir.iterdir():
            if file.name.startswith("input_"):
                target = tardis_dir / (basename + file.name[len("input"):])
            else:
                target = tardis_dir / file.name
            if target.exists():
                target.unlink()
            file.rename(target)
        prediction_dir.rmdir()  # remove the now-empty Predictions directory
