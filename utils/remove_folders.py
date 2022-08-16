import shutil
from pathlib import Path
from paths import DATA_DIR


def remove_folders(relative_paths):
    """Remove Folders. Used to initialize folder."""

    base_path = Path(DATA_DIR)
    paths = [base_path / path for path in relative_paths]
    for path in paths:
        string_path = path.absolute().as_posix()

        if path.is_dir() or path.is_file():
            print(f"Remove {string_path}")
            shutil.rmtree(path)
        else:
            print(f"(Does not exist) Skip to remove {string_path}")
