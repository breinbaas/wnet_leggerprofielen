from leveelogic.helpers import case_insensitive_glob
from pathlib import Path

ERROR_PATH = r"D:\Development\Python\wnet_leggerprofielen\data\output\errors"


with open(f"{ERROR_PATH}/errors.csv", "w") as f_out:
    for file in case_insensitive_glob(ERROR_PATH, ".stix"):
        try:
            error_filename = f"{file.stem}.error"
            msg = open(Path(ERROR_PATH) / error_filename).readlines()[0]
            f_out.write(f"{file.stem}.stix; {msg}\n")
            # print(msg)
        except Exception as e:
            print(f"Error reading file '{file.stem}'; '{e}'")
