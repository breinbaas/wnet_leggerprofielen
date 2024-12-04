from leveelogic.helpers import case_insensitive_glob
from leveelogic.objects.levee import Levee, AnalysisType
from pathlib import Path
from tqdm import tqdm
import shutil, glob, os


from settings import (
    PATH_ALL_STIX_FILES,
    PATH_STIX_BISHOP,
    PATH_STIX_LIFTVAN,
    PATH_STIX_SPENCER,
    PATH_STIX_ERRORS,
)


# clean output paths
for p in [PATH_STIX_BISHOP, PATH_STIX_LIFTVAN, PATH_STIX_SPENCER, PATH_STIX_ERRORS]:
    files = glob.glob(f"{p}/*.stix")
    for f in files:
        os.remove(f)


stix_files = case_insensitive_glob(PATH_ALL_STIX_FILES, ".stix")

for stix_file in tqdm(stix_files):
    try:
        levee = Levee.from_stix(Path(stix_file), x_reference_line=0.0)
        if levee.analysis_type == AnalysisType.BISHOP_BRUTE_FORCE:
            fname = Path(PATH_STIX_BISHOP) / stix_file.name
        elif levee.analysis_type == AnalysisType.SPENCER_GENETIC:
            fname = Path(PATH_STIX_SPENCER) / stix_file.name
        elif levee.analysis_type == AnalysisType.UPLIFT_VAN_PARTICLE_SWARM:
            fname = Path(PATH_STIX_LIFTVAN) / stix_file.name
        else:
            fname = Path(PATH_STIX_ERRORS) / stix_file.name
    except:
        fname = Path(PATH_STIX_ERRORS) / stix_file.name

    shutil.copy(str(stix_file), str(fname))
