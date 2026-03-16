#!/usr/bin/env python3
import argparse
from pathlib import Path

import pycolmap

from . import (
    extract_features,
    logger,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
)
from .pairs_from_retrieval import main as pairs_from_retrieval


PAIRING_METHODS = ("retrieval", "exhaustive")
MAPPER_TYPES = ("incremental", "global")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a basic HLOC mapping pipeline on a folder of images."
    )
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--matching_method",
        choices=PAIRING_METHODS,
        default="retrieval",
        help="How to generate SfM image pairs.",
    )
    parser.add_argument(
        "--feature_type",
        choices=sorted(
            name
            for name, conf in extract_features.confs.items()
            if not conf["output"].startswith("global-feats-")
        ),
        default="superpoint_aachen",
    )
    parser.add_argument(
        "--matcher_type",
        choices=sorted(match_features.confs.keys()),
        default="superglue",
    )
    parser.add_argument(
        "--mapper_type",
        "--mapper-type",
        choices=MAPPER_TYPES,
        default="global",
        help="COLMAP SfM backend to use during reconstruction.",
    )
    parser.add_argument("--num_matched", type=int, default=50)
    parser.add_argument("--camera_model", default="OPENCV")
    parser.add_argument("--single_camera", action="store_true", default=True)
    parser.add_argument("--no_single_camera", dest="single_camera", action="store_false")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Best-effort resume: reuse existing intermediate files when present.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_image_list(image_dir: Path):
    extensions = {".jpg", ".png"}
    references = sorted(
        p.relative_to(image_dir).as_posix()
        for p in image_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in extensions
    )
    if not references:
        raise ValueError(f"No images found in {image_dir}.")
    return references


def main():
    args = parse_args()

    image_dir = args.image_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    references = get_image_list(image_dir)

    feature_conf = extract_features.confs[args.feature_type]
    matcher_conf = match_features.confs[args.matcher_type]

    features = out_dir / f"{feature_conf['output']}.h5"
    matches = out_dir / f"{matcher_conf['output']}.h5"
    sfm_pairs = out_dir / (
        "pairs-exhaustive.txt"
        if args.matching_method == "exhaustive"
        else "pairs-retrieval.txt"
    )
    sfm_dir = out_dir / "sfm"
    sfm_outputs = (
        sfm_dir / "images.bin",
        sfm_dir / "cameras.bin",
        sfm_dir / "points3D.bin",
    )

    if args.resume and features.exists():
        logger.info("Reusing existing local features at %s.", features)
    else:
        extract_features.main(
            feature_conf,
            image_dir,
            image_list=references,
            feature_path=features,
        )

    if args.resume and sfm_pairs.exists():
        logger.info("Reusing existing image pairs at %s.", sfm_pairs)
    elif args.matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    else:
        retrieval_conf = extract_features.confs["netvlad"]
        retrieval_path = out_dir / f"{retrieval_conf['output']}.h5"
        if args.resume and retrieval_path.exists():
            logger.info("Reusing existing retrieval descriptors at %s.", retrieval_path)
        else:
            retrieval_path = extract_features.main(
                retrieval_conf,
                image_dir,
                export_dir=out_dir,
                image_list=references,
            )
        num_matched = min(len(references), args.num_matched)
        pairs_from_retrieval(
            retrieval_path,
            sfm_pairs,
            num_matched=num_matched,
            query_list=references,
            db_list=references,
        )

    if args.resume and matches.exists():
        logger.info("Reusing existing matches at %s.", matches)
    else:
        match_features.main(
            matcher_conf,
            sfm_pairs,
            features=features,
            matches=matches,
        )

    image_options = {"camera_model": args.camera_model}
    camera_mode = (
        pycolmap.CameraMode.SINGLE
        if args.single_camera
        else pycolmap.CameraMode.PER_IMAGE
    )

    if args.resume and all(path.exists() for path in sfm_outputs):
        logger.info("Reusing existing SfM model at %s.", sfm_dir)
        return

    reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=image_dir,
        pairs=sfm_pairs,
        features=features,
        matches=matches,
        camera_mode=camera_mode,
        image_options=image_options,
        mapper_type=args.mapper_type,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
