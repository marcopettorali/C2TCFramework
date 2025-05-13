# Copyright (C) 2025  Marco Pettorali

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import Counter
import hashlib
from multiprocessing import Process
import pickle
from pathlib import Path

import numpy as np
from networking.channel_model import e, override_channel_model_path
from networking.environment import Environment
from utils.geom import generate_random_points_in_polygon
from utils.printing import print
import shapely as sh
import multiprocessing as mp


def _find_closest_br_to_point_job(point, brs, is_crossing):
    closest_br = None
    closest_dist = float("inf")
    for br in brs:
        dist = br.pos.distance(point)
        if dist < closest_dist and not is_crossing(sh.LineString([br.pos, point])):
            closest_dist = dist
            closest_br = br

    if closest_br is None:
        raise Exception("No BR in LoS", point)

    return closest_br.label


def compute_process_closestbr_map(env: Environment, process: Process, samples=10000):
    if len(env.deployment) == 0:
        raise Exception("Deployment is empty")

    brs = env.deployment
    aoi_polygon = process.aoi.polygon

    # sample points in the aoi and compute the closest br that is in the LoS
    points = generate_random_points_in_polygon(aoi_polygon, samples)

    # call the job in parallel for each point
    is_crossing = env.is_crossing_obstacle

    with mp.Pool(mp.cpu_count()) as pool:
        closest_brs = pool.starmap(
            _find_closest_br_to_point_job,
            [(point, brs, is_crossing) for point in points],
        )

    # count the number of times each br is the closest to a point and normalize
    closest_br_labels_counter = Counter(closest_brs)

    closest_br_map = {br: count / len(closest_brs) for br, count in closest_br_labels_counter.items()}

    # sort the map by the closest br label
    closest_br_map = dict(sorted(closest_br_map.items()))

    return closest_br_map


def compute_avg_packet_loss(process, env: Environment, nsamples=1000):
    aoi = process.aoi.polygon
    random_samples = generate_random_points_in_polygon(aoi, nsamples)

    # for each sample compute the distance to all the BRs with LoS
    brs_positions = [br.pos for br in env.deployment]

    packet_loss_sum = 0

    for sample in random_samples:

        # filter out the BRs that are not in LoS
        los_brs = [br for br in brs_positions if not env.is_crossing_obstacle(sh.LineString([br, sample]))]

        # compute the distance to all the BRs
        distances = [br.distance(sample) for br in los_brs]

        # compute the packet loss probability
        packet_loss = np.prod([1 - e(d) for d in distances])

        packet_loss_sum += packet_loss

    # normalize the sum to compute the average packet loss
    packet_loss_sum /= nsamples
    return packet_loss_sum


def compute_brcoverage_matrix(*args, **kwargs):

    context = kwargs["context"]
    channel_model = kwargs["channel_model"]

    override_channel_model_path(channel_model)

    # ASSUMPTION: the BR coverage is influenced by:
    # - the environment (obstacles)
    # - the deployment of the BRs
    # - the AoIs of the processes
    # - the channel model used
    
    cache_material = [context["environment"], [x.aoi.polygon for x in context["processes"].values()], channel_model]
    cache_material = pickle.dumps(cache_material)
    cache_key = hashlib.sha1(str(cache_material).encode()).hexdigest()

    # check if the cache exists
    cache_file = Path(__file__).parent / "brcoverage_matrix_cache" / f"{cache_key}.pkl"
    if cache_file.exists():
        print(f"Loading cached coverage matrix from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    else:
        print(f"Cached coverage matrix NOT found ({cache_file})")

    matrix = {}
    for process_name in context["processes"]:
        process_row = compute_process_closestbr_map(context["environment"], context["processes"][process_name])

        packet_loss_probability = compute_avg_packet_loss(context["processes"][process_name], context["environment"])

        matrix[process_name] = {}
        for br in context["environment"].deployment:
            if br.label not in process_row:
                matrix[process_name][br.label] = 0
            else:
                # scale the value such that the sum of the row + packet loss is 1
                matrix[process_name][br.label] = float(process_row[br.label] * (1 - packet_loss_probability))

        # normalize back to 1
        row_sum = sum(matrix[process_name].values())
        matrix[process_name] = {br: value / row_sum for br, value in matrix[process_name].items()}

    # save the cache
    print(f"Saving cached coverage matrix to {cache_file}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(matrix, f)

    return matrix
