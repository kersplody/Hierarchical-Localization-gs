#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        description="Run a basic HLOC mapping pipeline on a folder of images.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--image-path",
        "--image_path",
        dest="image_path",
        type=Path,
        required=True,
        help="Directory containing the input images.",
    )
    parser.add_argument(
        "--database-path",
        "--database_path",
        dest="database_path",
        type=Path,
        required=True,
        help="Path to the COLMAP database file. Its parent directory stores HLOC intermediate files.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        type=Path,
        required=True,
        help="Directory where reconstructed COLMAP models are exported as numbered subdirectories.",
    )
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
    parser.add_argument(
        "--mapper-option",
        "--mapper_option",
        dest="mapper_options_raw",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Set nested mapper options as key=value. For global mapping, prefer "
            "paths like mapper.min_num_matches=15 or "
            "mapper.bundle_adjustment.ceres.solver_options.max_num_iterations=200."
        ),
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
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def parse_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def set_nested_option(options: Dict[str, Any], path: List[str], value: Any):
    target = options
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def merge_nested_options(target: Dict[str, Any], source: Dict[str, Any]):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_nested_options(target[key], value)
        else:
            target[key] = value


def convert_colmap_global_mapper_option(flag: str, value: Any) -> Tuple[List[str], Any]:
    if not flag.startswith("GlobalMapper."):
        raise ValueError(f"Unsupported COLMAP flag `{flag}`.")
    name = flag.split(".", 1)[1]

    direct_mapper_options = {
        "min_num_matches",
        "ignore_watermarks",
        "num_threads",
        "random_seed",
        "decompose_relative_pose",
        "ba_num_iterations",
        "skip_rotation_averaging",
        "skip_track_establishment",
        "skip_global_positioning",
        "skip_bundle_adjustment",
        "skip_retriangulation",
        "track_intra_image_consistency_threshold",
        "track_required_tracks_per_view",
        "track_min_num_views_per_track",
        "max_angular_reproj_error_deg",
        "max_normalized_reproj_error",
        "min_tri_angle_deg",
        "ba_skip_fixed_rotation_stage",
        "ba_skip_joint_optimization_stage",
    }
    if name in direct_mapper_options:
        return ["mapper", name], value

    if name.startswith("gp_"):
        if name == "gp_max_num_iterations":
            if hasattr(pycolmap.GlobalPositionerOptions(), "max_num_iterations"):
                return ["mapper", "global_positioning", "max_num_iterations"], value
        return ["mapper", "global_positioning", name[3:]], value

    if name.startswith("ba_ceres_"):
        suffix = name[len("ba_ceres_") :]
        if suffix == "max_num_iterations":
            return [
                "mapper",
                "bundle_adjustment",
                "ceres",
                "solver_options",
                "max_num_iterations",
            ], value
        return ["mapper", "bundle_adjustment", "ceres", suffix], value

    if name.startswith("ba_"):
        return ["mapper", "bundle_adjustment", name[3:]], value

    if name.startswith("tri_"):
        return ["mapper", "retriangulation", name[4:]], value

    if name.startswith("ra_"):
        return ["mapper", "rotation_averaging", name[3:]], value

    return ["mapper", name], value


def parse_extra_args(extra_args: List[str], mapper_type: str):
    mapper_options: Dict[str, Any] = {}
    verbose_level = 2

    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg == "--v":
            if i + 1 >= len(extra_args):
                raise ValueError("Expected a value after `--v`.")
            verbose_level = int(extra_args[i + 1])
            i += 2
            continue
        if not arg.startswith("--"):
            raise ValueError(f"Unexpected argument `{arg}`.")
        if i + 1 >= len(extra_args):
            raise ValueError(f"Expected a value after `{arg}`.")

        flag = arg[2:]
        value = parse_scalar(extra_args[i + 1])
        if mapper_type != "global":
            raise ValueError(
                f"COLMAP-style mapper flags are only supported with the global mapper, got `{arg}`."
            )
        path, mapped_value = convert_colmap_global_mapper_option(flag, value)
        set_nested_option(mapper_options, path, mapped_value)
        i += 2

    return mapper_options, verbose_level


def parse_mapper_option_entry(entry: str, mapper_type: str) -> Dict[str, Any]:
    idx = entry.find("=")
    if idx == -1:
        raise ValueError(
            f"Invalid --mapper-option `{entry}`. Expected KEY=VALUE."
        )

    key = entry[:idx]
    value = parse_scalar(entry[idx + 1 :])

    if not key:
        raise ValueError(f"Invalid --mapper-option `{entry}`. Empty option key.")

    if mapper_type == "global" and key.startswith("GlobalMapper."):
        path, mapped_value = convert_colmap_global_mapper_option(key, value)
    else:
        path = key.split(".")
        if any(not part for part in path):
            raise ValueError(
                f"Invalid --mapper-option `{entry}`. Empty path component in `{key}`."
            )
        if mapper_type == "global" and path[0] != "mapper":
            raise ValueError(
                "Global mapper options passed via --mapper-option must start with "
                "`mapper.` to match the pycolmap option tree."
            )
        mapped_value = value

    options: Dict[str, Any] = {}
    set_nested_option(options, path, mapped_value)
    return options


def default_mapper_options(mapper_type: str):
    if mapper_type != "global":
        return None

    options: Dict[str, Any] = {
        "mapper": {
            "bundle_adjustment": {
                "ceres": {
                    "solver_options": {
                        "max_num_iterations": 100,
                    }
                }
            }
        }
    }
    if hasattr(pycolmap.GlobalPositionerOptions(), "max_num_iterations"):
        options["mapper"]["global_positioning"] = {"max_num_iterations": 50}
    return options


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
    args, extra_args = parse_args()
    mapper_options = default_mapper_options(args.mapper_type) or {}
    for entry in args.mapper_options_raw:
        merge_nested_options(
            mapper_options,
            parse_mapper_option_entry(entry, args.mapper_type),
        )
    extra_mapper_options, verbose_level = parse_extra_args(extra_args, args.mapper_type)
    merge_nested_options(mapper_options, extra_mapper_options)
    if args.verbose:
        pycolmap.logging.verbose_level = max(pycolmap.logging.verbose_level, 2)
    if extra_args:
        pycolmap.logging.verbose_level = max(pycolmap.logging.verbose_level, verbose_level)

    image_dir = args.image_path
    database_path = args.database_path
    output_path = args.output_path
    work_dir = database_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    references = get_image_list(image_dir)

    feature_conf = extract_features.confs[args.feature_type]
    matcher_conf = match_features.confs[args.matcher_type]

    features = work_dir / f"{feature_conf['output']}.h5"
    matches = work_dir / f"{matcher_conf['output']}.h5"
    sfm_pairs = work_dir / (
        "pairs-exhaustive.txt"
        if args.matching_method == "exhaustive"
        else "pairs-retrieval.txt"
    )
    sfm_dir = work_dir / "sfm"
    has_existing_models = any(
        path.is_dir() and path.name.isdigit() for path in output_path.iterdir()
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
        retrieval_path = work_dir / f"{retrieval_conf['output']}.h5"
        if args.resume and retrieval_path.exists():
            logger.info("Reusing existing retrieval descriptors at %s.", retrieval_path)
        else:
            retrieval_path = extract_features.main(
                retrieval_conf,
                image_dir,
                export_dir=work_dir,
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

    if args.resume and has_existing_models:
        logger.info("Reusing existing exported model(s) at %s.", output_path)
        return

    reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=image_dir,
        pairs=sfm_pairs,
        features=features,
        matches=matches,
        database_path=database_path,
        camera_mode=camera_mode,
        image_options=image_options,
        mapper_options=mapper_options or None,
        mapper_type=args.mapper_type,
        model_dir=output_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
