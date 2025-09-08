from resourceallocation.moera import MOERA
from resourceallocation.djnecora import DJNecora
import copy
import random

NUM_REPETITIONS = 30

context = DJNecora.load_context_from_file("configs/scenario1.json")

moera = MOERA()
djnecora = DJNecora("no_splitting", "first_fit")

moera.set_context(copy.deepcopy(context))
djnecora.set_context(copy.deepcopy(context))

for rep in range(NUM_REPETITIONS):
    print(f"--- REPETITION {rep + 1}/{NUM_REPETITIONS} ---")
    
    # get process list
    process_list = list(moera.context.processes.keys())

    # get MNs list
    mns_list = [ p for p in process_list for _ in range (0, context.processes[p].mns) ]

    # shuffle process list
    random.shuffle(mns_list)

    # transform each entry of the shuffled list in (process, i), where i is the index of the MN relative to process from 0 to N
    # e.g. [P0, P1, P0, P2, P1] -> [(P0,0), (P1,0), (P0,1), (P2,0), (P1,1)]
    mns_list = [(p, sum(1 for x in mns_list[:i] if x == p)) for i, p in enumerate(mns_list)]


    
    
    
