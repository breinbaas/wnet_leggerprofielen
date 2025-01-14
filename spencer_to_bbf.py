from leveelogic.helpers import case_insensitive_glob
from leveelogic.objects.levee import Levee, AnalysisType
from geolib.models.dstability import DStabilityModel
from tqdm import tqdm
from geolib.models.dstability.dstability_model import PersistablePoint
from pathlib import Path

INPUT_PATH = "data/input/stix/all"
OUTPUT_PATH = "data/input/stix/converted"

stix_files = case_insensitive_glob(INPUT_PATH, ".stix")

for file in tqdm(stix_files):
    levee = Levee().from_stix(file)

    if levee.analysis_type == AnalysisType.SPENCER_GENETIC:
        levee.spencer_to_bishop()
        filename = Path(file).stem
        levee.to_stix(Path(OUTPUT_PATH) / f"{filename}.bff.stix")
