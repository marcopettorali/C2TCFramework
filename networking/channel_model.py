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

import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from rich import console
from scipy.integrate import quad
from scipy.optimize import bisect
from scipy.special import binom
from scipy.stats import norm

console = console.Console()

print = console.print

#################################
# PARAMETERS
#################################

P = {
    "MODEL_PATH_OVERRIDE": None,
    "BANDWIDTH_HZ": 2 * 10**6,
    "TEMPERATURE_C": 20.0,
    "POWER_TRANSMITTER_MW": 1,
    "THERMAL_NOISE_CONSTANT_DBM": -95,
    "ANTENNA_GAIN_DB": 0,
    "RSSI_SENSITIVITY_DBM": -100,
    "SINR_THRESHOLD_DBM": -8,
    "FADING_SIGMA": 3.6,
    "PACKET_LENGTH_BYTES": 50,
    "MODEL_MAX_DISTANCE_M": 200,
    "MODEL_STEP_M": 0.5,
}

VERBOSE = True

#################################
# UTILS
#################################

BASE_FOLDER = Path(__file__).parent / "channel_models"


vprint = print if VERBOSE else lambda *args: None


CACHED_E = None


def override_channel_model_path(path):
    global P
    P["MODEL_PATH_OVERRIDE"] = path

    print(f"Model path override set to {path}")


def mw_to_dbm(x):
    return 10 * math.log10(x)


def dbm_to_mw(x):
    return 10 ** (x / 10)


#################################
# MODEL CORE
#################################


def path_loss(distance, s):
    return min(1, 1 / (dbm_to_mw(37.7) * (distance) ** 3.3 * dbm_to_mw(s)))


def power_received(distance, s, power_transmitter):
    gain = min(1, dbm_to_mw(P["ANTENNA_GAIN_DB"]) ** 2 * path_loss(distance, s))
    power = power_transmitter * gain

    if mw_to_dbm(power) < P["RSSI_SENSITIVITY_DBM"]:
        return 0
    else:
        return power


def snr(distance, s, noise_power):
    snr = power_received(distance, s, P["POWER_TRANSMITTER_MW"]) / noise_power
    if snr == 0 or (snr) < P["SINR_THRESHOLD_DBM"]:
        return 0
    else:
        return snr


def bit_error_rate(sinr):
    sum = 0
    for k in range(2, 16 + 1):
        sum += ((-1) ** k) * binom(16, k) * (math.e ** (20 * sinr * (-1 + 1 / k)))
    return sum / 30


def packet_error_rate(distance, s):
    ber = bit_error_rate(snr(distance, s, dbm_to_mw(P["THERMAL_NOISE_CONSTANT_DBM"])))
    packet_lenght_bits = P["PACKET_LENGTH_BYTES"] * 8
    return 1 - (1 - ber) ** packet_lenght_bits


def success_probability(distance, s=0):
    return 1 - packet_error_rate(distance, s)


#################################


def e(distance):
    if distance == 0:
        return 1

    global CACHED_E

    if CACHED_E is not None:
        return CACHED_E(distance)

    # load the model when this function is called for the first time
    _model_import_export()
    return CACHED_E(distance)


def e_freeze_job(distance):
    return quad(
        lambda x: success_probability(distance, x) * norm(loc=0, scale=P["FADING_SIGMA"]).pdf(x),
        -99,
        99,
    )[0]


def e_freeze(data=None, return_data=False):
    # precompute the values of e(x) in the range [0, max_distance]. This is useful to speed up the computation of e(x).
    # returns a callable function that interpolates the precomputed values.

    if data is None:
        xs = np.arange(P["MODEL_STEP_M"], P["MODEL_MAX_DISTANCE_M"], P["MODEL_STEP_M"])

        # compute the values of e(x) for each x in xs using multiprocessing
        vprint(
            f"Computing e_frozen with {len(xs)} samples in the range [{P['MODEL_STEP_M']} m, {P['MODEL_MAX_DISTANCE_M']} m) (step = {P['MODEL_STEP_M']} m) with {mp.cpu_count()} cores..."
        )
        with mp.Pool(mp.cpu_count()) as pool:
            ys = pool.map(e_freeze_job, xs)

        vprint("e_frozen computed successfully.")
    else:
        xs = data["xs"]
        ys = data["ys"]

    def e_frozen(x):
        if x < 0:
            raise ValueError("x must be greater than 0")
        if x >= xs[-1]:
            vprint(f"Warning: d={x}m is greater than the maximum distance. The result may be inaccurate.")

        return np.interp(x, xs, ys)

    if return_data:
        return e_frozen, {"xs": xs.tolist(), "ys": ys}
    else:
        return e_frozen


#################################
# HANDLE FILE EXPORT
#################################


def e_inverse(probability, min=0.0001, max=200, epsilon=0.0001):
    return bisect(lambda x, _: e(x) - probability, min, max, epsilon)


def _params_to_hashkey():
    key = json.dumps(P)
    hash_obj = hashlib.md5(key.encode())
    hex_hash = hash_obj.hexdigest()
    return hex_hash


def _model_import_export():
    global CACHED_E

    if P["MODEL_PATH_OVERRIDE"] is not None:
        filename = BASE_FOLDER / P["MODEL_PATH_OVERRIDE"]

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found at MODEL_PATH_OVERRIDE={filename}")

        # parameters are not saved in the file, so delete them from P (except for MODEL_PATH_OVERRIDE)
        val = P["MODEL_PATH_OVERRIDE"]
        P.clear()
        P["MODEL_PATH_OVERRIDE"] = val
    else:
        filename = f"{BASE_FOLDER}/{_params_to_hashkey()}.json"

    # check if the file already exists, load it
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            CACHED_E = e_freeze(data=data)
            vprint(f"Channel model loaded from '{filename}'")
    else:

        print(f"Channel model not found at {filename}. Generating...")

        model, data_xsys = e_freeze(return_data=True)
        CACHED_E = model
        data = {}
        data["params"] = P
        data["xs"] = data_xsys["xs"]
        data["ys"] = data_xsys["ys"]
        # append params to data
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Exported channel model to {filename}")


if __name__ == "__main__":
    xs = [
        1,
        2,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
    ]
    ys = [e(x) for x in xs]

    print(ys)

    import matplotlib.pyplot as plt

    plt.plot(xs, ys)
    plt.show()
