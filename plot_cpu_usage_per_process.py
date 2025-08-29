from resourceallocation.djnecora import DJNecora
from networking.entities import Host, Process
from utils.logging import info, print

SCENARIO = "scenario1"
MAX_MNS = 13
context = DJNecora.load_context_from_file(f"configs/{SCENARIO}.json")

processes_data = {}
host: Host = context.hosts["BR0"]

process: Process
for process_name, process in context.processes.items():
    max_delay_ms = process.max_delay_ms

    for (p, h, c, m), g in context.links["gamma_tot_precomputed"].items():
        mns = 10
        if p == process_name and h == host.label and g <= max_delay_ms and m == mns:
            print(f"Process: {p}, Host: {h}, CPU Share: {c:<.2f}, CPU GHz: {c * host.cpu_ghz:<.2f}, MNs: {m}, Delay: {g:<.2f}")
    exit()

    for mns in range(1, MAX_MNS + 1):

        # find min CPU share to satisfy those MNs
        best_fit_record = min(
            (
                (p, h, c, m)
                for (p, h, c, m), g in context.links["gamma_tot_precomputed"].items()
                if p == process_name and h == host.label and g <= max_delay_ms and m == mns
            ),
            key=lambda x: x[2],  # minimize CPU share that guarantees MNs
            default=None,
        )
        cpu_share = best_fit_record[2] * host.cpu_ghz if best_fit_record else None

        processes_data[process_name] = processes_data.get(process_name, {})
        processes_data[process_name][mns] = cpu_share

print(processes_data)
