"""
MOERA solves this optimization problem at each time slot:
    min     E_O + E_Q + E_R + E_M
    s.t.    \sum_{s} x_{s,u,t} >= \lambda_u
            \sum_{u} x_{s,u,t} <= C_s
            x_{s,u,t} >= 0

where
    x       =   amount of CPU allocated for an app on a node
    E_O     =   Operation cost = energy OR CPU usage OR mantainance cost
    E_Q     =   Service quality cost = network delay. It includes routing from access node to the node hosting the app
    E_R     =   Reconfiguration cost = cost associated with incresing CPU usage on nodes (e.g. powering up a new server)
    E_M     =   Migration cost = cost associated with migrating an app from one node to another
    \lambda =   CPU demand of an app
    C       =   CPU capacity of a node

Adaptations we made to MOERA to compare it with DJ-NECORA:
- 1 user = 1 MN
- Similarly to DJ-NECORA, the task of 1 MN is served by a single edge node
  Hence, we add another constraint:
    \exists! s' : x_{s',u,t} > 0 \forall u,t
- MOERA executes at each time slot ==> every minute. This means that in some time slots nothing changes, while in others new apps/MNs arrive.
  This behavior is similar to DJ-NECORA, which is triggered by events (new app/MN arrival) when the time slot is very short (1 minute should be ok).
- For E_O, we use CPU usage for fair comparison with DJ-NECORA
- For E_Q, MOERA assumes delays to be constant, and only considers network delays. Hence, we use the average delay of the distribution we use for DJ-NECORA.
- For E_R, we set it to 0 (we simply do not consider it in DJ-NECORA)
- For E_M, we set it to 0 (we simply do not consider it in DJ-NECORA).
- Moreover, since we do not consider migration, once an app is placed on a node, it will not be moved.
  Hence, we add another constraint:
    x_{s,u,t} = x_{s,u,t-1} \forall s,u,t
- MOERA needs the positions of the MNs at each time slot, and does not assume to have any AOIs/future positions in advance.
  Since we do not consider migration, we also consider a static scenario.
  Otherwise, mobility without migration can be very limiting for MOERA, since once an app is placed on a node, it will not be moved, even if the MN moves far away from the node.
- Since 1 user = 1 MN, we set the CPU demand of an app (\lambda) to be the minimum CPU GHz required to satisfy the app's latency requirement with the required reliability for a single MN.
  In this computation, we consider:
    - network delay = average network delay of the distribution we use for DJ-NECORA, averaged over all possible locations of the MN/edge node
    - no queuing delay (we are allocating only 1 MN)
    - execution time is variabile. We compute the CPU GHz required to satisfy the latency requirement with the required reliability, considering only the variability of the execution time.
"""
