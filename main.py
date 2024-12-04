import logging
import os
import glob
from pathlib import Path
import shutil
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from geolib.models.dstability import DStabilityModel


from leveelogic.helpers import case_insensitive_glob
from leveelogic.objects.levee import Levee
from leveelogic.objects.soil import Soil


from objects.dijktrajecten import Dijktrajecten
from objects.iposearch import IPOSearch
from objects.uitgangspunten import Uitgangspunten
from settings import *
from helpers import (
    stix_has_solution,
    get_natural_slopes_line,
    get_highest_pl_level,
    uplift_at,
    xs_at,
    z_at,
    points_between,
    move_to_error_directory,
)

# indien de volgende waarde True is dan worden alle berekeningen opnieuw gemaakt, ook als er al een oplossing is
FORCE_RECALCULATION = False

plt.switch_backend("agg")


#############
# BASISDATA #
#############
try:
    dijktrajecten = Dijktrajecten.from_csv(CSV_FILE_DTH, CSV_FILE_ONDERHOUDSDIEPTE)
    ipo_search = IPOSearch.from_csv(CSV_FILE_IPO)
except Exception as e:
    raise ValueError(
        "Kan de invoer niet lezen, zijn alle bestanden in settings.py gedefinieerd?"
    )


# maak het pad met de tijdelijke berekeningen leeg
files = glob.glob(f"{PATH_TEMP_CALCULATIONS}/*.stix")
for f in files:
    os.remove(f)

# maak het pad met de foute berekeningen leeg
files = glob.glob(f"{PATH_ERRORS}/*.stix")
for f in files:
    os.remove(f)

# aan de slag!

stix_files = case_insensitive_glob(PATH_ALL_STIX_FILES, ".stix")

for stix_file in stix_files:
    # maak een aparte directory per berekening
    base_path = Path(PATH_DEBUG) / stix_file.stem
    base_path.mkdir(parents=True, exist_ok=True)

    files = glob.glob(f"{base_path}/*")
    for f in files:
        os.remove(f)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=str(base_path / f"00_{stix_file.stem}.log"),
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,  # INFO in production
    )

    # is er al een oplossing?
    if not FORCE_RECALCULATION and stix_has_solution(stix_file):
        logging.warning(
            "Er is al een oplossing gevonden, deze berekening wordt niet opnieuw gedaan."
        )
        move_to_error_directory(
            stix_file,
            "Er is al een oplossing gevonden, deze berekening wordt niet opnieuw gedaan.",
        )
        continue

    #################
    # READ THE FILE #
    #################
    try:
        levee = Levee.from_stix(Path(stix_file), x_reference_line=0.0)
        dm = DStabilityModel()
        dm.parse(Path(stix_file))
    except Exception as e:
        logging.error(f"Cannot open file '{stix_file.stem}.stix' got error; {e}")
        move_to_error_directory(
            stix_file, f"Cannot open file '{stix_file.stem}.stix' got error; {e}"
        )
        continue

    ##############################
    # EXTRACT INFO FROM FILENAME #
    ##############################
    args = stix_file.stem.split("_")
    try:
        dtcode = args[0]
        chainage = float(args[1])
    except Exception as e:
        logging.error(
            f"Kan de naam van het dijktraject en/of de metrering niet uit de bestandsnaam '{stix_file.stem}' bepalen'",
        )
        move_to_error_directory(
            stix_file,
            f"Kan de naam van het dijktraject en/of de metrering niet uit de bestandsnaam '{stix_file.stem}' bepalen'",
        )
        continue

    # copy the original calculation into the base path
    shutil.copy(stix_file, base_path / f"01_original.stix")

    # write the converted levee to stix for comparison
    levee.to_stix(base_path / f"02_as_levee_object.stix")

    ##################################
    # GET THE CALCULATION PARAMETERS #
    ##################################
    try:
        dth = dijktrajecten.get_by_code(dtcode).dth_2024_at(chainage)
        river_level = dijktrajecten.get_by_code(dtcode).mhw_2024_at(chainage)
        onderhoudsdiepte = dijktrajecten.get_by_code(dtcode).onderhoudsdiepte_at(
            chainage
        )
    except Exception as e:
        logging.error(
            f"Fout bij het zoek naar informatie voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Fout bij het zoek naar informatie voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    if dth is None:
        logging.error(
            f"Geen DTH kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Geen DTH kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    if river_level is None:
        logging.error(
            f"Geen MHW kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Geen MHW kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    if onderhoudsdiepte is None:
        logging.error(
            f"Geen onderhoudsdiepte kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Geen onderhoudsdiepte kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    ipo = ipo_search.get_ipo(dtcode, chainage)
    if ipo is None:
        logging.error(
            f"Geen IPO informatie gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Geen IPO informatie gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    try:
        required_sf = IPO_DICT[ipo]
    except Exception as e:
        logging.error(
            f"Ongeldige IPO informatie '{ipo}' gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Ongeldige IPO informatie '{ipo}' gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    try:
        model_factor = MODELFACTOR[levee.analysis_type]
    except Exception as e:
        logging.error(
            f"Ongeldig analyse type '{levee.analysis_type}' gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Ongeldig analyse type '{levee.analysis_type}' gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    # vertaal naar een object
    uitgangspunten = Uitgangspunten(
        river_level=river_level,
        dth=dth,
        onderhoudsdiepte=onderhoudsdiepte,
        ipo=ipo,
        required_sf=required_sf,
        kruinbreedte=CREST_WIDTH,
        pl_surface_offset=PL_SURFACE_OFFSET,
        traffic_load_width=TRAFFIC_LOAD_WIDTH,
        traffic_load_magnitude=TRAFFIC_LOAD_MAGNITUDE,
        schematiseringsfactor=SCHEMATISERINGSFACTOR,
        modelfactor=MODELFACTOR[levee.analysis_type],
    )

    logging.info(
        f"Aangehouden uitgangspunten voor dijktraject '{dtcode}' metrering {chainage}:"
    )
    logging.info(f"\tIPO klasse: {uitgangspunten.ipo}")
    logging.info(f"\tVereiste veiligheidsfactor: {uitgangspunten.required_sf}")
    logging.info(f"\tDijktafelhoogte: {uitgangspunten.dth}")
    logging.info(f"\tMaatgevend hoogwater: {uitgangspunten.river_level}")
    logging.info(f"\tOnderhoudsdiepte: {uitgangspunten.onderhoudsdiepte}")
    logging.info(f"\tKruinbreedte voor minimaal profiel: {uitgangspunten.kruinbreedte}")
    logging.info(
        "\tPolderniveau voor minimaal profiel: gelijk aan laagste punt op maaiveld dat geen onderdeel is van de sloot"
    )
    logging.info(
        "\tDe oude verkeersbelasting wordt gehanteerd en elke laag met een cohesie die groter is dan 0 wordt op 50% consolidatie gezet"
    )
    logging.info(
        f"Minimale lengte glijvlak is ingesteld op {MIN_SLIP_PLANE_LENGTH} meter"
    )
    logging.info(
        f"Minimale diepte glijvlak is ingesteld op {MIN_SLIP_PLANE_DEPTH} meter"
    )
    logging.info(
        f"De afstand tussen de freatische lijn en het maaiveld is ingesteld op {PL_SURFACE_OFFSET} meter"
    )
    logging.info(
        f"NB. Er wordt GEEN rekening gehouden met het verloop van de stijghoogte naar de freatische lijn."
    )
    logging.info(
        f"Gebruikt ophoogmateriaal yd={OPH_YD}, ys={OPH_YS}, c={OPH_C}, phi={OPH_PHI}"
    )

    ##############################################
    # CHECK OF WE BBF, LIFTVAN OF SPENCER HEBBEN #
    ##############################################
    if not levee.analysis_type in MODELFACTOR.keys():
        logging.error(
            f"Dit model gebruikt het model {levee.analysis_type} waarvoor geen code beschikbaar is om de model factor voor te bepalen"
        )
        move_to_error_directory(
            stix_file,
            f"Dit model gebruikt het model {levee.analysis_type} waarvoor geen code beschikbaar is om de model factor voor te bepalen",
        )
        continue

    ########################################
    # WE ONDERSTEUNEN TOT NU TOE ENKEL BBF #
    ########################################
    if levee.analysis_type != AnalysisType.BISHOP_BRUTE_FORCE:
        logging.error(
            f"Dit bestand gebruikt model '{levee.analysis_type}' en we kunnen tot nu toe alleen met BBF omgaan"
        )
        move_to_error_directory(
            stix_file,
            f"Dit bestand gebruikt model '{levee.analysis_type}' en we kunnen tot nu toe alleen met BBF omgaan",
        )
        continue

    ####################
    # CHECK CURRENT SF #
    ####################
    try:
        dm.execute()
        org_sf = dm.get_result(0, 0).FactorOfSafety
        org_sf = org_sf / SCHEMATISERINGSFACTOR / MODELFACTOR[levee.analysis_type]
        logging.info(
            f"De huidige veiligheidsfactor uit de originele berekening bedraagt {org_sf:.2f}"
        )

        # voeg de verkeersbelasting aan de dijk toe
        loads = dm._get_loads(scenario_index=0, stage_index=0)
        if len(loads.UniformLoads) != 0:
            try:
                tl_width = loads.UniformLoads[0].End - loads.UniformLoads[0].Start
                tl_magnitude = loads.UniformLoads[0].Magnitude
                levee.add_traffic_load(CREST_WIDTH - tl_width, tl_width, tl_magnitude)
            except Exception as e:
                logging.warning(
                    f"De verkeersbelasting kan niet uitgelezen worden, foutmelding: {e}"
                )
        else:
            logging.warning("Er is geen verkeerslast gevonden in deze berekening!")

        levee.to_stix(base_path / f"03_as_levee_with_trafficload.stix")
        dsc = levee.calculate(calculation_name="levee")
        sf = round(dsc.get_model_by_name("levee").result.safety_factor, 2)
        sf = sf / SCHEMATISERINGSFACTOR / MODELFACTOR[levee.analysis_type]
        logging.info(
            f"De huidige veiligheidsfactor volgens de gestripte berekening bedraagt {sf:.2f}"
        )
        if abs(sf - org_sf) > 0.10:
            logging.warning(
                f"Er is een groot verschil aangetroffen tussen de originele ({org_sf:.2f}) en gestripte ({sf:.2f}) berekening. Dit duidt op een berekening waarin het effect van PL2/PL3 groot is. "
            )
    except Exception as e:
        logging.error(
            f"Fout bij het berekenen van de originele veiligheidsfactor; '{e}'",
        )
        move_to_error_directory(
            stix_file,
            f"Fout bij het berekenen van de originele veiligheidsfactor; '{e}'",
        )
        continue

    # voeg de grondsoort ophoogmateriaal toe
    levee.soils.append(
        Soil(
            code="Ophoogmateriaal",
            yd=OPH_YD,
            ys=OPH_YS,
            c=OPH_C,
            phi=OPH_PHI,
            color="#adacac",
        )
    )

    # voeg de constraints toe
    if levee.analysis_type == AnalysisType.BISHOP_BRUTE_FORCE:
        levee.add_bbf_constraints(
            min_slipplane_depth=MIN_SLIP_PLANE_DEPTH,
            min_slipplane_length=MIN_SLIP_PLANE_LENGTH,
        )

    levee.to_stix(base_path / f"04_iteration_start.stix")

    #############################
    # ITEREER TOT EEN OPLOSSING #
    #############################
    slope_factor = INITIAL_SLOPE_FACTOR
    iteration = 1
    counter = 5  # voor de opeenvolgende bestanden in de debug directory
    solution = None
    done = False
    while not done:
        try:
            logging.info(
                f"Generating and calculating iteration {iteration} with slope factor {slope_factor:.2f}"
            )
            # create a copy of the base levee
            levee_copy = deepcopy(levee)
            x1 = levee_copy.left
            z1 = uitgangspunten.dth
            x2 = (
                uitgangspunten.kruinbreedte
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
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot([p[0] for p in levee.surface], [p[1] for p in levee.surface], "k")
            ax.plot([p[0] for p in profile_line], [p[1] for p in profile_line], "r")
            ax.set_aspect("equal", adjustable="box")
            fig.savefig(
                Path(base_path)
                / f"{counter:02d}_{stix_file.stem}.profile_line_{slope_factor:.2f}.png"
            )
            counter += 1
            plt.clf()

            logging.debug(f"Points on cut line, {profile_line}")

            try:
                levee_copy._cut(profile_line)
            except Exception as e:
                logging.error(
                    f"Wegsnijden leidt tot een ongeldige geometrie, waarschijnlijk is de geometrie te krap gekozen"
                )
                raise e

            fill_line = [(0.0, z2), (uitgangspunten.kruinbreedte, z2)] + profile_line[
                2:
            ]

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
                    [(levee_copy.left, river_level), (levee_copy.right, river_level)]
                )
                p2 = (intersections[0][0], river_level)
            except Exception as e:
                logging.error(
                    f"Kan geen snijpunt vinden met het nieuwe maaiveld en de rivier waterstand van {river_level}"
                )
                raise e

            p3 = (0, uitgangspunten.river_level - 0.2)
            p4 = (uitgangspunten.kruinbreedte, uitgangspunten.river_level - 0.6)
            plline_points = [(x1, uitgangspunten.river_level), p2, p3, p4]

            if levee_copy.has_ditch:
                plline_points += [
                    (p[0], p[1] - PL_SURFACE_OFFSET)
                    for p in profile_line[2:]
                    if p[0] < xd1
                ]
                plline_points += [(xd1, polder_level), (levee_copy.right, polder_level)]
            else:
                plline_points += [
                    (p[0], p[1] - PL_SURFACE_OFFSET) for p in profile_line[2:]
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
                    CREST_WIDTH - TRAFFIC_LOAD_WIDTH,
                    TRAFFIC_LOAD_WIDTH,
                    TRAFFIC_LOAD_MAGNITUDE,
                )

            calculation_name = (
                f"{stix_file.stem}_iteration_{iteration}_slope_{slope_factor:.2f}.stix"
            )
            levee_copy.to_stix(
                Path(PATH_TEMP_CALCULATIONS) / f"{calculation_name}.stix"
            )
            levee_copy.to_stix(base_path / f"{counter:02d}_{calculation_name}")
            counter += 1
            try:
                dsc = levee_copy.calculate(calculation_name=calculation_name)
                sf = round(
                    dsc.get_model_by_name(calculation_name).result.safety_factor, 2
                )
                logging.info(f"De berekende veiligheidsfactor = {sf:.2f}")
                sf = sf / SCHEMATISERINGSFACTOR / MODELFACTOR[levee.analysis_type]
                logging.info(
                    f"Met modelfactor {MODELFACTOR[levee.analysis_type]} en schematisatie factor {SCHEMATISERINGSFACTOR} wordt de veiligheidsfactor {sf:.2f}"
                )
            except Exception as e:
                logging.error(
                    f"Could not calculate slope {slope_factor:.2f}, got error {e}"
                )
                raise e

                # update, we start with the natural slopes
                # if this leads to SF >= SF_REQUIRED we are done and have the solution
            if sf >= required_sf:  # and sf <= required_sf + SF_MARGIN:
                logging.info(
                    f"Found a solution after {iteration} iteration(s) with slope factor={slope_factor:.2f}"
                )
                solution = levee_copy
                try:
                    x_uittredepunt = dsc.get_model_by_name(
                        calculation_name
                    ).result.slipplane[-1][0]
                except Exception as e:
                    logging.error(f"kan uittredepunt niet bepalen, foutmelding: {e}")
                    continue

                done = True
            elif sf < required_sf:
                slope_factor *= 1.2
            else:
                slope_factor /= 1.1

            iteration += 1

            if not done and iteration > MAX_ITERATIONS:
                logging.error(
                    f"After {MAX_ITERATIONS} iterations we still have no solution, skipping this levee"
                )
                done = True
                break
        except Exception as e:
            logging.error(f"Onverwachte fout opgetreden '{e}'")
            try:
                levee_copy.to_stix(
                    Path(base_path)
                    / f"{counter:02d}_{stix_file.stem}_iteration_{iteration}_with_error.stix"
                )
                counter += 1
            except:
                logging.debug(
                    "Could not save error file, probably because it is impossible to generate the model"
                )
            iteration += 1
            done = iteration > MAX_ITERATIONS

    if solution is None:
        logging.error("Geen oplossing gevonden, controleer de bovenstaande log.")
        move_to_error_directory(
            stix_file,
            "Geen oplossing gevonden",
        )
        continue

    # get the remaining profile based on the slopes of the soils at x_uittredepunt
    # NOTE that we only use the soils directly under x_uittredepunt for the slopes
    # any changes in soil layers to the right of x_uittredepunt are ignored
    # TODO > can be optimized
    natural_slopes_line = get_natural_slopes_line(solution, x_uittredepunt)
    natural_slopes_line_left = [
        (-1 * p[0], p[1]) for p in get_natural_slopes_line(solution, 0)
    ]

    # create the final line
    final_line = natural_slopes_line_left[::-1]
    final_line += [p for p in solution.surface if p[0] >= 0.0 and p[0] < x_uittredepunt]
    final_line += natural_slopes_line

    # we moeten de lijn aan kunnen passen dus we moeten van de tuples af
    final_line = [[p[0], p[1]] for p in final_line]

    # determine if there is uplift
    # get the highest head from the original calculation at the bottom of the slope (x3)
    plmax = get_highest_pl_level(dm, x3)
    has_uplift = uplift_at(levee=levee_copy, x=x3, hydraulic_head=plmax)

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
        logging.error(
            f"De x coordinaat voor het punt onderhoudsdiepte - 2.0m voor overgang BZ naar BBZ aan waterzijde kan niet gevonden worden, melding: {e}"
        )
        move_to_error_directory(
            stix_file,
            f"De x coordinaat voor het punt onderhoudsdiepte - 2.0m voor overgang BZ naar BBZ aan waterzijde kan niet gevonden worden, melding: {e}",
        )
        continue

    x2_before_x1 = x2 <= x1

    p2 = (x2, z2)  # BBZ -> BZ
    try:
        x3 = xs_at(final_line, dth - 1.5)[0]
    except Exception as e:
        logging.error(
            f"De x coordinaat voor het punt DHT-1.5m voor overgang KZ naar BZ aan waterzijde kan niet gevonden worden, melding: {e}"
        )
        move_to_error_directory(
            stix_file,
            f"De x coordinaat voor het punt DHT-1.5m voor overgang KZ naar BZ aan waterzijde kan niet gevonden worden, melding: {e}",
        )
        continue
    p3 = (x3, dth - 1.5)
    p4 = (0.0, dth)

    z5 = z_at(final_line, x_uittredepunt)
    if z5 is None:
        logging.error("De z coordinaat op het uittredepunt kan niet gevonden worden")
        continue
    p5 = (x_uittredepunt, z5)

    has_excavation_intersection = True
    try:
        x6 = xs_at(final_line, polder_level - 2.0)[-1]
        p6 = (x6, polder_level - 2.0)
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

    for spg in levee.soilpolygons:
        soil = levee.get_soil_by_code(spg.soilcode)
        p = Polygon(spg.points, facecolor=soil.color)
        ax.add_patch(p)

    ax.plot([p[0] for p in levee.surface], [p[1] for p in levee.surface], "k")
    ax.plot([p[0] for p in final_line], [p[1] for p in final_line], "k--")
    ax.plot(
        [p[0] for p in levee.phreatic_line],
        [p[1] for p in levee.phreatic_line],
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
    # for debugging
    fig.savefig(base_path / f"{counter:02d}_{stix_file.stem}_solution.png")
    counter += 1

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
    lines = ["code,x,z\n"]
    for p in csv_points:
        lines.append(f"{p[0]},{p[1]:.2f},{p[2]:.2f}\n")

    with open(Path(PATH_SOLUTIONS_CSV) / f"{stix_file.stem}_solution.csv", "w") as f:
        for l in lines:
            f.write(l)

    # for debugging
    with open(base_path / f"{counter:02d}_{stix_file.stem}_solution.csv", "w") as f:
        for l in lines:
            f.write(l)
        counter += 1

    solution.to_stix(Path(PATH_SOLUTIONS) / f"{stix_file.stem}_solution.stix")
    # for debugging
    solution.to_stix(base_path / f"{counter:02d}_{stix_file.stem}_solution.stix")
    counter += 1

    # move the input file
    stix_file.replace(Path(PATH_SOLUTIONS) / stix_file.name)
    plt.close("all")
