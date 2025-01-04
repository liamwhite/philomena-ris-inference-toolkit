from PIL import Image
from pathlib import Path
import cli
import db
import dinov2
import fire
import subprocess
import tempfile
import torch
import threading
import traceback
import warnings


def _check_programs():
    import shutil

    for program in ["safe-rsvg-convert", "mediastat", "svgstat"]:
        if not shutil.which(program):
            raise OSError(f"{program} not available in $PATH")


def _stat_program(program_name: str, args: list[str]) -> dict:
    output = subprocess.check_output([program_name, *args]).decode("utf-8").strip()
    size, frames, width, height, num, den = [int(x) for x in output.split(" ")]
    return {
        "size": size,
        "frames": frames,
        "width": width,
        "height": height,
        "num": num,
        "den": den,
    }


def _check_dimensions(stat: dict):
    width = stat["width"]
    height = stat["height"]
    pixels = width * height
    if Image.MAX_IMAGE_PIXELS and pixels >= Image.MAX_IMAGE_PIXELS * 2:
        raise Image.DecompressionBombError("")


def _infer(path: str, model: torch.ScriptModule, device: str) -> torch.Tensor:
    with torch.no_grad():
        result = dinov2.get_model_result(path, model, device).tolist()
    torch.cuda.empty_cache()
    return result


def _process_record(path: str, model: torch.ScriptModule, device: str) -> torch.Tensor:
    p = Path(path)

    if p.suffix == ".svg":
        _check_dimensions(_stat_program("svgstat", [path]))
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png") as preview:
            subprocess.check_call(
                ["safe-rsvg-convert", path, preview.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return _infer(preview.name, model, device)
    elif p.suffix in [".webm", ".gif"]:
        stat = _stat_program("mediastat", [path])
        seek = stat["num"] / stat["den"] / 2
        _check_dimensions(stat)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png") as preview:
            subprocess.check_call(
                ["mediathumb", path, str(seek), preview.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return _infer(preview.name, model, device)
    else:
        return _infer(path, model, device)


def _process_all_records(
    database: str, model: torch.ScriptModule, device: str, lock: threading.Lock
):
    conn = db.connect(database_path=database)

    for id, path in db.get_images_for_process(conn, lock):
        try:
            features = _process_record(path, model, device)
            db.pass_image_process(conn, id, features, lock)
            print(f"\r{path}    ", end="")
        except Image.DecompressionBombError:
            continue
        except (OSError, ValueError, SyntaxError):
            db.fail_image_process(conn, id, lock)
            print(f"\nImage {id} is corrupt")
        except Exception:
            traceback.print_exc()


def _update_image_limit(pixels: int):
    Image.MAX_IMAGE_PIXELS = pixels


def batch(
    model_path: str,
    device: str,
    concurrency: int,
    pixels: int,
    database: str,
    base_path: str | None = None,
):
    """
    Run batch inference on unprocessed images in the database.
    :param model_path: path to the serialized DINOv2 model
    :param device: PyTorch device to use, `cpu` or `cuda`
    :param concurrency: number of concurrent image decoding threads
    :param pixels: maximum number of pixels allowed to decode
    :param database: path to the database
    :param base_path: path from which relative file paths are read/written
    """
    torch.set_default_device(device)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    cli.force_interrupt_shutdown()
    cli.set_base_path(base_path)
    db.reset_in_progress_process(database_path=database)
    _check_programs()
    _update_image_limit(pixels)

    model = torch.jit.load(model_path).eval().to(device)
    lock = threading.Lock()
    threads = []

    for _ in range(0, concurrency):
        thread = threading.Thread(
            target=_process_all_records, args=(database, model, device, lock)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print("\nDone")


if __name__ == "__main__":
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
    fire.Fire()
