import logging
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pydantic import BaseModel
from typing import List, Tuple, Optional

from leveelogic.objects.levee import Levee
from geolib.models.dstability import DStabilityModel


from helpers import (
    get_natural_slopes_line,
    write_to_debug_directory,
    move_to_error_directory,
    get_highest_pl_level,
    uplift_at,
    xs_at,
    z_at,
    points_between,
)
from settings import (
    MAX_ITERATIONS,
    PATH_TEMP_CALCULATIONS,
    PATH_SOLUTIONS_CSV,
    PATH_SOLUTIONS,
    PATH_SOLUTIONS_PLOTS,
)
from objects.uitgangspunten import Uitgangspunten


class Solution(BaseModel):
    levee: Levee
    x_uittredepunt: float
    x3: float
    dth: float
    polder_level: float
    profile_line: List[Tuple[float, float]]


def iterate_solution(
    stix_file: Path, levee: Levee, slope_factor: float, uitgangspunten: Uitgangspunten
) -> Optional[Solution]:
    iteration = 1
    done = False
    while not done:
        try:
            logging.info(
                f"Generating and calculating iteration {iteration} with slope {slope_factor:.2f}"
            )
            # create a copy of the base levee
            levee_copy = deepcopy(levee)
            x1 = levee_copy.left
            z1 = uitgangspunten.dth
            x2 = (
                uitgangspunten.crest_width
            )  # we expect the reference line to be on x=0.0, TODO > check?
            z2 = uitgangspunten.dth
            x4 = levee_copy.right

            # we zetten het maaiveld in de polder gelijk aan het laagste punt op
            # het maaiveld dat geen onderdeel is van een eventuele aanwezige sloot
            if len(levee_copy.ditch_points) == 4:
                surface_points = [
                    p
                    for p in levee_copy.surface
                    if p[0] > 0
                    and p[0] < levee_copy.ditch_points[0][0]
                    or p[0] > levee_copy.ditch_points[-1][0]
                ]
            else:
                surface_points = [p for p in levee_copy.surface if p[0] > 0]

            #

            z4 = min(p[1] for p in surface_points)

            natural_slopes = get_natural_slopes_line(
                levee_copy, x2, slope_factor=slope_factor
            )
            slope_points = [(x2, z2)]
            for i in range(1, len(natural_slopes)):
                p1x, p1z = natural_slopes[i - 1]
                p2x, p2z = natural_slopes[i]
                if p1z >= z4 and z4 >= p2z:
                    z = z4
                    x = p1x + (p1z - z) / (p1z - p2z) * (p2x - p1x)
                    slope_points.append((x, z))
                    break

                slope_points.append(natural_slopes[i])

            profile_line = [(x1, z1)] + slope_points + [(x4, z4)]
            x3, z3 = slope_points[-1]

            # if we have a ditch add it to the new profile
            if levee_copy.has_ditch:
                dp = levee_copy.ditch_points

                # check if the ditch bottom is above z3, if so skip the ditch
                ditch_zmin = min([p[1] for p in dp])
                if ditch_zmin >= z3:
                    break  # no need to add the ditch because the original bottom is above the new polderlevel

                # get the original slopes
                slope_left = (dp[1][0] - dp[0][0]) / (dp[0][1] - dp[1][1])
                slope_right = (dp[-1][0] - dp[-2][0]) / (dp[-1][1] - dp[-2][1])
                # and the original width
                ditch_width = dp[2][0] - dp[1][0]

                # if we pass the original ditch we need to move the ditch
                if x3 > dp[0][0] - 0.1:
                    xs = x3 + 0.1
                else:
                    xs = dp[0][0]

                xd1 = xs
                zd1 = z3
                zd2 = dp[1][1]
                xd2 = xd1 + slope_left * (zd1 - zd2)
                xd3 = xd2 + ditch_width
                zd3 = dp[2][1]
                zd4 = z4
                xd4 = xd3 + slope_right * (zd4 - zd3)

                # check if we did not pass x4, if so adjust x4
                if xd4 >= x4:
                    x4 = xd4 + 0.1

                profile_line = profile_line[:3] + [
                    (xd1, zd1),
                    (xd2, zd2),
                    (xd3, zd3),
                    (xd4, zd4),
                    (x4, z4),
                ]

            # create a plot for debugging purposes
            # fig, ax = plt.subplots(figsize=(15, 5))
            # ax.plot([p[0] for p in dm.surface], [p[1] for p in dm.surface], "k")
            # ax.plot([p[0] for p in profile_line], [p[1] for p in profile_line], "r")
            # ax.set_aspect("equal", adjustable="box")
            # fig.savefig(Path(PLOT_PATH) / f"{stix_file.stem}.profile_line_{slope:.2f}.png")

            # create the model
            # new_levee = deepcopy(levee)

            logging.debug(f"Points on cut line, {profile_line}")

            try:
                levee_copy._cut(profile_line)
            except Exception as e:
                raise ValueError(
                    "Wegsnijden leidt tot een ongeldige geometrie, waarschijnlijk is de geometrie te krap gekozen"
                )

            fill_line = [(0.0, z2), (uitgangspunten.crest_width, z2)] + profile_line[2:]

            logging.debug(f"Points on fill line, {fill_line}")
            levee_copy._fill(
                fill_line=fill_line,
                soilcode="Ophoogmateriaal",
            )

            # Slootpeil gelijk aan dat in de sloot of maaiveld minus 0.15m
            if levee_copy.has_ditch:
                polder_level = levee_copy.phreatic_level_at(
                    levee_copy.ditch_points[1][0]
                )
                # update 6-8-2024, het kan voorkomen dat het slootpeil hoger ligt dan het laagste punt van het maaiveld
                # in dit geval leggen we het slootpeil op maaiveld minus 0.15m
                polder_level = min(polder_level, z4 - 0.15)
                # logging.info(f"Het peil in de sloot is gebaseerd op het originele slootpeil eventueel gecorrigeerd voor een lager liggend maaiveld en bedraagt {polder_level:.2f}")
            else:
                polder_level = z4 - 0.15
                # logging.info(f"Er is geen sloot informatie gevonden, voor het waterpeil in de polder wordt het laatste punt van het maaiveld minus 0.15m gebruikt wat neerkomt op {polder_level:.2f}")

            # generate the phreatic line
            # Updated 6-8-2024
            # point 2 = intersection with surface
            try:
                intersections = levee_copy.get_surface_intersections(
                    [
                        (levee_copy.left, uitgangspunten.river_level),
                        (levee_copy.right, uitgangspunten.river_level),
                    ]
                )
                p2 = (intersections[0][0], uitgangspunten.river_level)
            except Exception as e:
                raise ValueError(
                    f"Kan geen snijpunt vinden met het nieuwe maaiveld en de rivier waterstand van {uitgangspunten.river_level}"
                )

            p3 = (0, uitgangspunten.river_level - 0.2)
            p4 = (uitgangspunten.crest_width, uitgangspunten.river_level - 0.6)
            # point 3 = 0, river_level - 0.2
            # point 4 = CREST_WIDTH, river_level - 0.6
            plline_points = [(x1, uitgangspunten.river_level), p2, p3, p4]

            if levee_copy.has_ditch:
                plline_points += [
                    (p[0], p[1] - uitgangspunten.pl_surface_offset)
                    for p in profile_line[2:]
                    if p[0] < xd1
                ]
                plline_points += [(xd1, polder_level), (levee_copy.right, polder_level)]
            else:
                plline_points += [
                    (p[0], p[1] - uitgangspunten.pl_surface_offset)
                    for p in profile_line[2:]
                ]

            # check that points going to the right are below the previous point
            final_pl_points = [plline_points[0]]
            for p in plline_points[1:]:
                y_prev = final_pl_points[-1][1]
                if p[1] > y_prev:
                    final_pl_points.append((p[0], y_prev))
                else:
                    final_pl_points.append(p)
            levee_copy.add_phreatic_line(final_pl_points)

            if not levee_copy.has_traffic_load:
                levee_copy.add_traffic_load(
                    uitgangspunten.crest_width - uitgangspunten.traffic_load_width,
                    uitgangspunten.traffic_load_width,
                    uitgangspunten.traffic_load_magnitude,
                )

            # # try and copy the PL line settings
            # # first check if it is possible, it is NOT possible if a reference line
            # # intersects the new surface
            # adjust_rllines = True
            # for rl in dm.waternets[0].ReferenceLines:
            #     points = [(p.X, p.Z) for p in rl.Points]
            #     if len(polyline_polyline_intersections(profile_line, points)) > 0:
            #         logging.warning(
            #             f"Referentie lijn (id='{rl.Id}') kruist met de nieuwe geometrie waardoor het niet mogelijk is om de waterspanningen over te nemen. Dit heeft effect op de berekende stabiliteitsfactor."
            #         )
            #         adjust_rllines = False
            #         break

            # if adjust_rllines:
            #     # get the referenceline information
            #     pass
            #     # copy the headlines and remember their old and new id

            #     # add the referencelines with the new ids

            calculation_name = (
                f"{stix_file.stem}_iteration_{iteration}_slope_{slope_factor:.2f}"
            )
            levee_copy.to_stix(
                Path(PATH_TEMP_CALCULATIONS) / f"{calculation_name}.stix"
            )

            try:
                dsc = levee_copy.calculate(calculation_name=calculation_name)
                sf = round(
                    dsc.get_model_by_name(calculation_name).result.safety_factor, 2
                )
                logging.info(f"De berekende veiligheidsfactor = {sf:.2f}")
                sf = (
                    sf
                    / uitgangspunten.schematiseringsfactor
                    / uitgangspunten.modelfactor
                )
                logging.info(
                    f"Met modelfactor {uitgangspunten.modelfactor} en schematisatie factor {uitgangspunten.schematiseringsfactor} wordt de veiligheidsfactor {sf:.2f}"
                )
            except Exception as e:

                raise ValueError(
                    f"Could not calculate slope {slope_factor:.2f}, got error {e}"
                )

            # update, we start with the natural slopes
            # if this leads to SF >= SF_REQUIRED we are done and have the solution
            if sf >= uitgangspunten.required_sf:  # and sf <= required_sf + SF_MARGIN:
                logging.info(
                    f"Found a solution after {iteration} iteration(s) with slope factor={slope_factor:.2f}"
                )
                try:
                    x_uittredepunt = dsc.get_model_by_name(
                        calculation_name
                    ).result.slipplane[-1][0]
                except Exception as e:
                    raise ValueError(f"kan uittredepunt niet bepalen, foutmelding: {e}")

                return Solution(
                    levee=levee_copy,
                    x_uittredepunt=x_uittredepunt,
                    x3=x3,
                    dth=uitgangspunten.dth,
                    polder_level=polder_level,
                    profile_line=profile_line,
                )

            elif sf < uitgangspunten.required_sf:
                slope_factor *= 1.2
            else:
                slope_factor /= 1.1

            iteration += 1

            if not done and iteration > MAX_ITERATIONS:
                move_to_error_directory(
                    stix_file=stix_file,
                    message=f"No solution after {MAX_ITERATIONS} iterations.",
                )
                return None

        except Exception as e:
            try:
                write_to_debug_directory(
                    levee,
                    stix_file,
                    str(e),
                )
            except:
                logging.debug(
                    "Could not save error file, probably because it is impossible to generate the model"
                )
            iteration += 1
            done = iteration > MAX_ITERATIONS


def generate_output(
    stix_file: Path,
    original_model: DStabilityModel,
    original_levee: Levee,
    solution: Solution,
    onderhoudsdiepte: float,
):
    natural_slopes_line = get_natural_slopes_line(
        solution.levee, solution.x_uittredepunt
    )
    natural_slopes_line_left = [
        (-1 * p[0], p[1]) for p in get_natural_slopes_line(solution.levee, 0)
    ]

    # create the final line
    final_line = natural_slopes_line_left[::-1]
    final_line += [
        p
        for p in solution.levee.surface
        if p[0] >= 0.0 and p[0] < solution.x_uittredepunt
    ]
    final_line += natural_slopes_line

    # we moeten de lijn aan kunnen passen dus we moeten van de tuples af
    final_line = [[p[0], p[1]] for p in final_line]

    # determine if there is uplift
    # get the highest head from the original calculation at the bottom of the slope (x3)
    plmax = get_highest_pl_level(original_model, solution.x3)
    has_uplift = uplift_at(levee=original_levee, x=solution.x3, hydraulic_head=plmax)

    # find the characteristic points
    p1 = final_line[0]  # BBZ eind rivierzijde
    z2 = onderhoudsdiepte - 2.0
    try:
        if final_line[0][1] < z2:
            x2 = xs_at(final_line, z2)[0]
        else:
            slope = (final_line[1][0] - final_line[0][0]) / (
                final_line[1][1] - final_line[0][1]
            )
            x2 = final_line[0][0] - (final_line[0][1] - z2) * slope
    except Exception as e:
        raise ValueError(
            f"De x coordinaat voor het punt onderhoudsdiepte - 2.0m voor overgang BZ naar BBZ aan waterzijde kan niet gevonden worden, melding: {e}"
        )

    x1 = original_levee.left
    x2_before_x1 = x2 <= x1

    p2 = (x2, z2)  # BBZ -> BZ
    try:
        x3 = xs_at(final_line, solution.dth - 1.5)[0]
    except Exception as e:
        raise ValueError(
            f"De x coordinaat voor het punt DHT-1.5m voor overgang KZ naar BZ aan waterzijde kan niet gevonden worden, melding: {e}"
        )
    p3 = (x3, solution.dth - 1.5)
    p4 = (0.0, solution.dth)

    z5 = z_at(final_line, solution.x_uittredepunt)
    if z5 is None:
        raise ValueError("De z coordinaat op het uittredepunt kan niet gevonden worden")

    p5 = (solution.x_uittredepunt, z5)

    has_excavation_intersection = True
    try:
        x6 = xs_at(final_line, solution.polder_level - 2.0)[-1]
        p6 = (x6, solution.polder_level - 2.0)
        p7 = final_line[-1]  # BBZ eind landzijde
        points_to_plot = [p1, p2, p3, p4, p5, p6, p7]
    except Exception as e:
        # bakje snijdt met pleistoceen -> geen BBZ en einde ligt op 10m voorbij einde BZ
        final_line[-1][0] = p5[0] + 10.0
        p7 = final_line[-1]
        points_to_plot = [p1, p2, p3, p4, p5, p7]
        logging.warning(
            f"De x coordinaat voor het punt polderpeil-2.0m voor overgang BZ naar BBZ aan landzijde kan niet gevonden worden, melding: {e}"
        )
        logging.warning(
            "Aanname dat dit komt omdat er geen snijpunt gevonden kan worden omdat onderzijde bakje lager ligt dan de Pleistocene zandlaag."
        )
        has_excavation_intersection = False

    # plot the solution
    fig, ax = plt.subplots(figsize=(15, 5))

    for spg in solution.levee.soilpolygons:
        soil = solution.levee.get_soil_by_code(spg.soilcode)
        p = Polygon(spg.points, facecolor=soil.color)
        ax.add_patch(p)

    ax.plot(
        [p[0] for p in solution.levee.surface],
        [p[1] for p in solution.levee.surface],
        "k",
    )
    ax.plot([p[0] for p in final_line], [p[1] for p in final_line], "k--")
    ax.plot(
        [p[0] for p in solution.levee.phreatic_line],
        [p[1] for p in solution.levee.phreatic_line],
        "b",
    )

    # add the zones
    zmin = min([p[1] for p in final_line]) - 1.0
    zmax = max([p[1] for p in final_line]) + 1.0
    ax.plot([p1[0], p1[0]], [zmin, zmax], "k--")
    ax.scatter(
        [p[0] for p in points_to_plot],
        [p[1] for p in points_to_plot],
    )

    if x2_before_x1:
        ax.plot([p1[0], p1[0]], [zmin, zmax], "k--")
        ax.text(p1[0], zmax, "Start, BZ")
        ax.text(p1[0], p1[1], "25")
    else:
        ax.text(p1[0], zmax, "Start, BBZ")
        ax.text(p1[0], p1[1], "25")
        ax.plot([p2[0], p2[0]], [zmin, zmax], "k--")
        ax.text(p2[0], zmax, "BZ")
        ax.text(p2[0], p2[1], "25")
    ax.plot([p3[0], p3[0]], [zmin, zmax], "k--")
    ax.text(p3[0], zmax, "KZ")
    ax.text(p3[0], p3[1], "25")
    ax.text(p4[0], p4[1], "90")
    ax.plot([p5[0], p5[0]], [zmin, zmax], "k--")
    ax.text(p5[0], zmax, "BZ")
    ax.text(p5[0], p5[1], "25")
    if not has_uplift and has_excavation_intersection:
        ax.plot([p6[0], p6[0]], [zmin, zmax], "k--")
        ax.text(p6[0], zmax, "BBZ")
        ax.text(p6[0], p6[1], "25")
    ax.plot([p7[0], p7[0]], [zmin, zmax], "k--")
    ax.text(p7[0], zmax, "Einde")
    ax.text(p7[0], p7[1], "25")

    fig.savefig(Path(PATH_SOLUTIONS_PLOTS) / f"{stix_file.stem}_solution.png")

    # create the points with codes
    csv_points = []
    csv_points.append((25, *p1))
    if x2_before_x1:
        csv_points += [(99, *p) for p in points_between(final_line, p1[0], p3[0])]
    else:
        csv_points += [(99, *p) for p in points_between(final_line, p1[0], p2[0])]
        csv_points.append((25, *p2))
        csv_points += [(99, *p) for p in points_between(final_line, p2[0], p3[0])]
    csv_points.append((25, *p3))
    csv_points += [(99, *p) for p in points_between(final_line, p3[0], p4[0])]
    csv_points.append((90, *p4))
    csv_points += [(99, *p) for p in points_between(final_line, p4[0], p5[0])]
    csv_points.append((25, *p5))
    if not has_uplift and has_excavation_intersection:
        csv_points += [(99, *p) for p in points_between(final_line, p5[0], p6[0])]
        csv_points.append((25, *p6))
        csv_points += [(99, *p) for p in points_between(final_line, p6[0], p7[0])]
        csv_points.append((25, *p7))
    else:
        csv_points += [(99, *p) for p in points_between(final_line, p5[0], p7[0])]
        csv_points.append((25, *p7))

    # round
    csv_points = [(p[0], round(p[1], 2), round(p[2], 2)) for p in csv_points]

    # write a csv file
    with open(Path(PATH_SOLUTIONS_CSV) / f"{stix_file.stem}_solution.csv", "w") as f:
        f.write("code,x,z\n")
        for p in csv_points:
            f.write(f"{p[0]},{p[1]:.2f},{p[2]:.2f}\n")

    solution.levee.to_stix(Path(PATH_SOLUTIONS) / f"{stix_file.stem}_solution.stix")

    plt.clf()
