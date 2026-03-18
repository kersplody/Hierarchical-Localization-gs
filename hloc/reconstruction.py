import argparse
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pycolmap
import tqdm
from packaging import version

from . import logger
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)


MapperType = Literal["incremental", "global"]
GLOBAL_MAPPER_MIN_VERSION = version.parse("4.0.0")
MODEL_FILENAMES = (
    "images.bin",
    "cameras.bin",
    "points3D.bin",
    "frames.bin",
    "rigs.bin",
)


def get_incremental_options():
    if hasattr(pycolmap, "IncrementalPipelineOptions"):
        return pycolmap.IncrementalPipelineOptions()
    return pycolmap.IncrementalMapperOptions()


def get_global_options():
    if hasattr(pycolmap, "GlobalPipelineOptions"):
        return pycolmap.GlobalPipelineOptions()
    return pycolmap.GlobalMapperOptions()


def check_global_mapper_support():
    found_version = getattr(pycolmap, "__version__", "unknown")
    if found_version != "dev" and version.parse(found_version) < GLOBAL_MAPPER_MIN_VERSION:
        raise RuntimeError(
            "Global COLMAP mapping requires pycolmap>=4.0.0, "
            f"but found pycolmap=={found_version}."
        )
    if not hasattr(pycolmap, "global_mapping"):
        raise RuntimeError(
            "The installed pycolmap build does not support global_mapping()."
        )


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    with pycolmap.Database.open(database_path) as _:
        pass


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_names=image_list or [],
            options=options,
        )


def get_image_ids(database_path: Path) -> Dict[str, int]:
    images = {}
    with pycolmap.Database.open(database_path) as db:
        images = {image.name: image.image_id for image in db.read_all_images()}
    return images


def incremental_mapping(
    database_path: Path,
    image_dir: Path,
    sfm_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> dict[int, pycolmap.Reconstruction]:
    num_images = pycolmap.Database.open(database_path).num_images()
    pbars = []

    def restart_progress_bar():
        if len(pbars) > 0:
            pbars[-1].close()
        pbars.append(
            tqdm.tqdm(
                total=num_images,
                desc=f"Reconstruction {len(pbars)}",
                unit="images",
                postfix="registered",
            )
        )
        pbars[-1].update(2)

    reconstructions = pycolmap.incremental_mapping(
        database_path,
        image_dir,
        sfm_path,
        options=options or {},
        initial_image_pair_callback=restart_progress_bar,
        next_image_callback=lambda: pbars[-1].update(1),
    )

    return reconstructions


def global_mapping(
    database_path: Path,
    image_dir: Path,
    sfm_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> dict[int, pycolmap.Reconstruction]:
    return pycolmap.global_mapping(
        database_path, image_dir, sfm_path, options=options or {}
    )


def calibrate_view_graph(database_path: Path):
    if not hasattr(pycolmap, "calibrate_view_graph"):
        logger.warning(
            "The installed pycolmap build does not support calibrate_view_graph(); "
            "continuing without view graph calibration."
        )
        return

    logger.info("Calibrating the view graph before global reconstruction...")
    if not pycolmap.calibrate_view_graph(database_path):
        logger.warning("View graph calibration did not succeed; continuing anyway.")


def run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
    mapper_type: MapperType = "incremental",
    model_dir: Optional[Path] = None,
    skip_view_graph_calibration: bool = False,
) -> pycolmap.Reconstruction:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    if options is None:
        options = {}
    options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}

    with OutputCapture(verbose):
        if mapper_type == "incremental":
            reconstructions = incremental_mapping(
                database_path, image_dir, models_path, options=options
            )
        elif mapper_type == "global":
            check_global_mapper_support()
            if skip_view_graph_calibration:
                logger.info("Skipping view graph calibration before global reconstruction.")
            else:
                calibrate_view_graph(database_path)
            reconstructions = global_mapping(
                database_path, image_dir, models_path, options=options
            )
        else:
            raise ValueError(f"Unsupported mapper type: {mapper_type}")

    if len(reconstructions) == 0:
        logger.error("Could not reconstruct any model!")
        return None
    logger.info(f"Reconstructed {len(reconstructions)} model(s).")

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(
        f"Largest model is #{largest_index} " f"with {largest_num_images} images."
    )

    if model_dir is None:
        for filename in MODEL_FILENAMES:
            source = models_path / str(largest_index) / filename
            target = sfm_dir / filename
            if target.exists():
                target.unlink()
            if source.exists():
                shutil.move(str(source), str(target))
    else:
        model_indices = {str(index) for index in reconstructions}
        model_dir.mkdir(parents=True, exist_ok=True)
        for child in model_dir.iterdir():
            if child.is_dir() and child.name.isdigit() and child.name not in model_indices:
                shutil.rmtree(child)

        for index in sorted(reconstructions):
            source_dir = models_path / str(index)
            target_dir = model_dir / str(index)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            for filename in MODEL_FILENAMES:
                source = source_dir / filename
                if source.exists():
                    shutil.move(str(source), str(target_dir / filename))
    return reconstructions[largest_index]


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    database_path: Optional[Path] = None,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
    mapper_options: Optional[Dict[str, Any]] = None,
    mapper_type: MapperType = "incremental",
    model_dir: Optional[Path] = None,
    skip_view_graph_calibration: bool = False,
) -> pycolmap.Reconstruction:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = database_path or (sfm_dir / "database.db")
    database.parent.mkdir(parents=True, exist_ok=True)
    if mapper_type == "global":
        check_global_mapper_support()

    logger.info(f"Writing COLMAP logs to {sfm_dir / 'colmap.LOG.*'}")
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / "colmap.LOG.")

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features)
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)
    reconstruction = run_reconstruction(
        sfm_dir,
        database,
        image_dir,
        verbose,
        mapper_options,
        mapper_type=mapper_type,
        model_dir=model_dir,
        skip_view_graph_calibration=skip_view_graph_calibration,
    )
    if reconstruction is not None:
        logger.info(
            f"Reconstruction statistics:\n{reconstruction.summary()}"
            + f"\n\tnum_input_images = {len(image_ids)}"
        )
    return reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
    )
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument(
        "--skip_view_graph_calibration",
        action="store_true",
        help="Skip calibrate_view_graph() before running the global mapper.",
    )
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mapper_type",
        type=str,
        default="incremental",
        choices=["incremental", "global"],
        help="Select the COLMAP SfM backend exposed by pycolmap.",
    )

    parser.add_argument(
        "--image_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(pycolmap.ImageReaderOptions().todict()),
    )
    parser.add_argument(
        "--mapper_options",
        nargs="+",
        default=[],
        help=(
            "List of key=value mapper backend options. Use pipeline option keys for "
            "the selected --mapper_type."
        ),
    )
    args = parser.parse_args().__dict__

    image_options = parse_option_args(
        args.pop("image_options"), pycolmap.ImageReaderOptions()
    )
    mapper_type = args.get("mapper_type", "incremental")
    if mapper_type == "global":
        check_global_mapper_support()
    mapper_default_options = (
        get_incremental_options() if mapper_type == "incremental" else get_global_options()
    )
    mapper_options = parse_option_args(
        args.pop("mapper_options"), mapper_default_options
    )

    main(**args, image_options=image_options, mapper_options=mapper_options)
