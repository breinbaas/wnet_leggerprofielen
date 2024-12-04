import os
import glob
from settings import *
from pathlib import Path
from tqdm import tqdm
import shutil

from leveelogic.objects.levee import Levee, AnalysisType
from leveelogic.helpers import case_insensitive_glob


# remove all files
for p in [
    PATH_TEMP_CALCULATIONS,
    PATH_ERRORS,
    PATH_DEBUG,
    PATH_SOLUTIONS_PLOTS,
    PATH_SOLUTIONS_CSV,
    PATH_SOLUTIONS,
    PATH_ALL_STIX_FILES,
]:
    files = glob.glob(os.path.join(p, "*"))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

# remove directories in debug
for path in sorted(
    Path(PATH_DEBUG).rglob("*"), key=lambda p: len(p.parts), reverse=True
):
    if path.is_dir() and not any(path.iterdir()):  # Check if the directory is empty
        path.rmdir()

# copy the original files to the all files directory
stix_files = case_insensitive_glob(PATH_ORIGINAL_FILES, ".stix")

for stix_file in tqdm(stix_files):
    try:
        levee = Levee.from_stix(Path(stix_file), x_reference_line=0.0)
        fname = Path(PATH_ALL_STIX_FILES) / stix_file.name
    except:
        fname = Path(PATH_STIX_ERRORS) / stix_file.name

    shutil.copy(str(stix_file), str(fname))
