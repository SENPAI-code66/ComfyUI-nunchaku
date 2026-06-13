import sys
import os
import comfy.utils

folder_paths = None
try:
    from comfy.cmd import folder_paths
    from comfy.model_downloader import get_filename_list, get_full_path_or_raise
except (ImportError, ModuleNotFoundError):
    folder_paths = sys.modules["folder_paths"]
    from folder_paths import get_filename_list, get_full_path_or_raise

get_filename_list = get_filename_list
get_full_path_or_raise = get_full_path_or_raise

def safe_load_torch_file(model_path, *args, **kwargs):
    """
    Safely load a PyTorch model weights file.
    Validates if the file is a git-lfs pointer or too small/corrupted,
    and handles SafetensorError by automatically deleting the corrupted file.
    """
    if not os.path.exists(model_path):
        return comfy.utils.load_torch_file(model_path, *args, **kwargs)

    file_size = os.path.getsize(model_path)
    if file_size < 1024 * 1024:  # Less than 1MB
        try:
            with open(model_path, "rb") as f:
                header = f.read(100)
            if b"version https://git-lfs" in header:
                # Try to remove the git-lfs pointer file so it can be re-downloaded
                try:
                    os.remove(model_path)
                    del_msg = " The git-lfs pointer file has been deleted so it can be re-downloaded."
                except Exception:
                    del_msg = f" Please delete the pointer file at '{model_path}' manually."
                raise ValueError(
                    f"The file '{os.path.basename(model_path)}' is a Git LFS pointer, not the actual model weights.{del_msg} "
                    "Make sure Git LFS is installed and run 'git lfs pull', or run the workflow again to download it."
                )
        except ValueError as val_err:
            raise val_err
        except Exception:
            pass

        # If it's just too small and not a git-lfs pointer
        try:
            os.remove(model_path)
            del_msg = " The corrupted file has been deleted so it can be re-downloaded."
        except Exception:
            del_msg = f" Please delete the corrupted file at '{model_path}' manually."
        raise ValueError(
            f"The file '{os.path.basename(model_path)}' is too small ({file_size} bytes) to be a valid model.{del_msg}"
        )

    try:
        return comfy.utils.load_torch_file(model_path, *args, **kwargs)
    except Exception as e:
        err_msg = str(e)
        if "SafetensorError" in err_msg or "deserializing" in err_msg or "incomplete metadata" in err_msg or "file not fully covered" in err_msg:
            try:
                os.remove(model_path)
                deleted_msg = f" The corrupted file at '{model_path}' has been automatically deleted so it can be re-downloaded on the next run."
            except Exception:
                deleted_msg = f" Please delete the corrupted file at '{model_path}' manually."
            raise RuntimeError(
                f"Failed to load model because the file was corrupted or incomplete.{deleted_msg} "
                f"Original error: {err_msg}"
            ) from e
        raise e

__all__ = ["get_filename_list", "get_full_path_or_raise", "folder_paths", "safe_load_torch_file"]
