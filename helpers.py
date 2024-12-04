from shapely import get_coordinates
from typing import List, Tuple, Dict, Optional
from shapely.geometry import (
    Point,
    LineString,
    MultiPoint,
    Polygon,
    MultiLineString,
    GeometryCollection,
)
from leveelogic.objects.levee import Levee
from geolib.models.dstability import DStabilityModel
import logging
from pathlib import Path

from settings import UNITWEIGHT_WATER, PATH_SOLUTIONS, PATH_ERRORS, PATH_DEBUG


def stix_has_solution(stix_file: Path) -> bool:
    """Check if we already have a solution

    Args:
        stix_file (Path): Path to the stix file

    Returns:
        bool: True if we have a solution
    """
    solution_file = Path(PATH_SOLUTIONS) / f"{stix_file.stem}_solution.stix"
    return solution_file.exists()


def move_to_error_directory(stix_file: Path, message: str) -> None:
    """Move the given stix file to the error location and write a textfile with the error

    Args:
        stix_file (Path): The file that has an error
        message (str): The message to write to the log and the added textfile
    """
    # logging.error(message)
    with open(Path(PATH_ERRORS) / f"{stix_file.stem}.error", "w") as f:
        f.write(message)
    Path(stix_file).rename(Path(PATH_ERRORS) / f"{stix_file.stem}.stix")


def move_to_solution_directory(stix_file: Path) -> None:
    """Move the given stix file to the error location and write a textfile with the error

    Args:
        stix_file (Path): The file that has an error
        message (str): The message to write to the log and the added textfile
    """
    Path(stix_file).rename(Path(PATH_SOLUTIONS) / f"{stix_file.stem}.stix")


def write_to_debug_directory(levee: Levee, stix_file: Path, message: str) -> None:
    """Move the given stix file to the debug location and write a textfile with the error

    Args:
        stix_file (Path): The file that has an error
        message (str): The message to write to the log and the added textfile
    """
    # logging.error(message)
    with open(Path(PATH_DEBUG) / f"{stix_file.stem}._cutting_error.error", "w") as f:
        f.write(message)

    levee.to_stix(Path(PATH_DEBUG) / f"{stix_file.stem}_cutting_error.stix")


def line_polygon_intersections(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    polygon_points: List[Tuple[float, float]],
):
    result = []
    line = LineString([p1, p2])
    pg = Polygon(polygon_points)
    intersections = pg.intersection(line)

    if type(intersections) == MultiPoint:
        result = [(g.x, g.y) for g in intersections.geoms]
    elif type(intersections) == LineString:
        result = get_coordinates(intersections).tolist()
    elif type(intersections) == Point:
        x, y = intersections.coords.xy
        result.append((x[0], y[0]))
    elif type(intersections) == MultiLineString:
        result = get_coordinates(intersections).tolist()
    elif type(intersections) == GeometryCollection:
        geoms = [g for g in intersections.geoms if type(g) != Point]
        result = get_coordinates(geoms).tolist()
    else:
        raise ValueError(f"Unimplemented intersection type '{type(intersections)}'")

    if len(result) == 0:
        return []

    return sorted(result, key=lambda x: x[1], reverse=True)


def line_line_intersections(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    line: List[Tuple[float, float]],
):
    result = []
    line_1 = LineString([p1, p2])
    line_2 = LineString(line)
    intersections = line_2.intersection(line_1)

    if type(intersections) == MultiPoint:
        result = [(g.x, g.y) for g in intersections.geoms]
    elif type(intersections) == LineString:
        result = get_coordinates(intersections).tolist()
    elif type(intersections) == Point:
        x, y = intersections.coords.xy
        result.append((x[0], y[0]))
    elif type(intersections) == MultiLineString:
        result = get_coordinates(intersections).tolist()
    elif type(intersections) == GeometryCollection:
        geoms = [g for g in intersections.geoms if type(g) != Point]
        result = get_coordinates(geoms).tolist()
    else:
        raise ValueError(f"Unimplemented intersection type '{type(intersections)}'")

    if len(result) == 0:
        return []

    return sorted(result, key=lambda x: x[1], reverse=True)


def get_natural_slopes_line(
    levee: Levee, x_uittredepunt: float, slope_factor: float = 1.0
) -> List[Tuple[float, float]]:
    result = [(x_uittredepunt, levee.z_at(x_uittredepunt))]
    x = x_uittredepunt

    # create a slopes dictionary
    slopes = {}
    for sl in levee.soilpolygons:
        soil = levee.get_soil_by_code(sl.soilcode)

        if sl.soilcode not in slopes.keys():
            if soil.ys < 12:
                slopes[sl.soilcode] = 6 * slope_factor
            elif soil.ys > 18:
                slopes[sl.soilcode] = 4 * slope_factor
            elif soil.c >= 3.0:
                slopes[sl.soilcode] = 3 * slope_factor
            else:
                slopes[sl.soilcode] = 4 * slope_factor

    all_intersections = []
    for spg in levee.soilpolygons:
        intersections = line_polygon_intersections(
            (x, levee.top + 1.0), (x, levee.bottom - 1.0), spg.points
        )

        if len(intersections) > 0 and len(intersections) % 2 == 0:
            for i in range(int(len(intersections) / 2)):
                all_intersections.append(
                    {
                        "top": intersections[i * 2][1],
                        "bottom": intersections[i * 2 + 1][1],
                        "slope": slopes[spg.soilcode],
                    }
                )

    # sort all intersections
    all_intersections = sorted(all_intersections, key=lambda x: x["top"], reverse=True)

    # voeg nu de punten toe obv de helling van de grondsoorten
    # px, pz = x, max([p["top"] for p in all_intersections])
    px = x
    for intersection in all_intersections[:-1]:
        dz = intersection["top"] - intersection["bottom"]
        px += dz * intersection["slope"]
        result.append((px, intersection["bottom"]))

    return result


def z_at(line: List[Tuple[float, float]], x: float) -> Optional[float]:
    for i in range(1, len(line)):
        x1, z1 = line[i - 1]
        x2, z2 = line[i]

        if x1 <= x and x <= x2:
            return z1 + (x - x1) / (x2 - x1) * (z2 - z1)

    return None


def xs_at(line: List[Tuple[float, float]], z: float) -> Optional[List[float]]:
    left = line[0][0]
    right = line[-1][0]
    return sorted([p[0] for p in line_line_intersections((left, z), (right, z), line)])


def points_between(
    line: List[Tuple[float, float]], x_start: float, x_end: float
) -> List[Tuple[float, float]]:
    return [p for p in line if x_start < p[0] and p[0] < x_end]


def get_highest_pl_level(dm: DStabilityModel, x: float) -> Optional[float]:
    wnet = dm._get_waternet(scenario_index=0, stage_index=0)
    pls = []
    for hl in wnet.HeadLines:
        for i in range(1, len(hl.Points)):
            p1 = hl.Points[i - 1]
            p2 = hl.Points[i]
            if p1.X <= x and x <= p2.X:
                pls.append(p1.Z + (x - p1.X) / (p2.X - p1.X) * (p2.Z - p1.Z))
                break

    if len(pls) > 0:
        return max(pls)
    return None


def uplift_at(
    levee: Levee, x: float, ym: float = 1.1, hydraulic_head: Optional[float] = None
) -> bool:
    logging.info("Calculating uplift...")
    all_intersections = []
    for spg in levee.soilpolygons:
        intersections = line_polygon_intersections(
            (x, levee.top + 1.0), (x, levee.bottom - 1.0), spg.points
        )

        if len(intersections) > 0 and len(intersections) % 2 == 0:
            for i in range(int(len(intersections) / 2)):
                all_intersections.append(
                    {
                        "top": intersections[i * 2][1],
                        "bottom": intersections[i * 2 + 1][1],
                        "soilcode": spg.soilcode,
                    }
                )

    # sort all intersections
    all_intersections = sorted(all_intersections, key=lambda x: x["top"], reverse=True)

    # bepaal de hoogte van de freatische lijn
    pl = levee.phreatic_level_at(x)
    if pl is None:  # er is geen freatische waterstand dus ook geen opdrijven
        logging.warning("Geen freatische waterstand gevonden dus geen opdrijven.")
        return False

    if hydraulic_head is None:
        hh = pl
    else:
        hh = hydraulic_head

    sigmav = 0.0
    for i in range(len(all_intersections)):
        top = all_intersections[i]["top"]
        bottom = all_intersections[i]["bottom"]
        soil = levee.get_soil_by_code(all_intersections[i]["soilcode"])
        soilcode = soil.code.lower()

        if (
            soilcode.startswith("z")
            or soilcode.startswith("pleist")
            or soilcode.startswith("material_z")
        ):
            logging.info(
                f"De {i+1}de/ste laag vanaf het maaiveld heet '{soilcode}' en wordt als watervoerende laag voor de opdrijfberekening gehanteerd."
            )

            if i == 0:  # toplaag is zand, geen opdrijven
                logging.warning("De toplaag is zand dus geen opdrijven.")
                return False

            up = (hh - top) * UNITWEIGHT_WATER

            logging.info(
                f"Opwaartse druk={up:.2f}, gewicht grondlagen / 1.1={(sigmav / ym):.2f}"
            )
            result = up >= (sigmav / ym)
            if result:
                logging.info("Opdrijven geconstateerd.")
                return True
            else:
                logging.info("Geen opdrijven geconstateerd.")
                return False

        if top > pl and pl > bottom:
            sigmav += (top - pl) * soil.yd
            sigmav += (pl - bottom) * soil.ys
        elif top <= pl:
            sigmav += (top - bottom) * soil.ys
        else:
            sigmav += (top - bottom) * soil.yd

        logging.info(
            f"Laag van {top:.2f} tot {bottom:.2f} toegevoegd, sigma;v={sigmav:.2f}"
        )

    # no sand layer found
    logging.warning("Geen zandlaag gevonden")
    return False
