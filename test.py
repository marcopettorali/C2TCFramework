from resourceallocation.context import load_context
from resourceallocation.jnecora import _compute_end_to_end_communication_delays, compute_delay_at_min_reliability
from utils.logging import LogLevel, debug, set_logging_level


context = load_context(filename="configs/scenario1.json")
process = context.processes["P7"]
host = context.hosts["BR5"]
process.mns = 50

context = _compute_end_to_end_communication_delays(context)

debug("<dummy computation>")
cpu_share = 0.04
set_logging_level(LogLevel.NONE)
compute_delay_at_min_reliability(context, process, host, cpu_share)

set_logging_level(LogLevel.ALL)
cpu_share = 0.02
import time
start = time.time()
compute_delay_at_min_reliability(context, process, host, cpu_share)
debug(f"Computed delay at min reliability took: {time.time() - start:.2f} seconds")