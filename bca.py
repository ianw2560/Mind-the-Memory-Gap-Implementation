import numpy as np

def pick_Bopt(batch_sizes, T_values, L_values, SLO, eps=0.1, tol=1e-3):
    B = np.array(batch_sizes) # Batch size
    T = np.array(T_values) # Throughput
    L = np.array(L_values) # Latency

    idx1 = np.where(B == 1)[0]
    if idx1.size > 0:
        T1 = T[idx1[0]]
    else:
        T1 = T[0]

    eff_ratio = T / (B * T1 + 1e-12)
    feasible_mask = (L <= SLO) & (eff_ratio > eps)

    # Satisfy latency and throughput conditions
    feasible_B = B[feasible_mask]
    feasible_T = T[feasible_mask]
    feasible_L = L[feasible_mask]

    if feasible_B.size == 0:
        return {
            'Bopt': None,
            'T': None,
            'L': None,
            'feasible_list': []
        }

    maxT = feasible_T.max()
    near_mask = (feasible_T >= maxT * (1 - tol))
    near_B = feasible_B[near_mask]
    near_T = feasible_T[near_mask]
    near_L = feasible_L[near_mask]

    pick_idx = np.argmin(near_B)
    Bopt = int(near_B[pick_idx])
    Topt = float(near_T[pick_idx])
    Lopt = float(near_L[pick_idx])

    feasible_list = sorted(
        [{'B': int(b), 'T': float(t), 'L': float(l)} 
         for b,t,l in zip(feasible_B, feasible_T, feasible_L)],
        key=lambda x: x['B']
    )

    return {
        'Bopt': Bopt,
        'T': Topt,
        'L': Lopt,
        'feasible_list': feasible_list
    }


if __name__ == "__main__":
    batch_sizes = [1,2,4,8,16,32,64,96,128,256,512]
    T_values = [100,190,360,680,1200,2000,3000,3200,3300,3350,3360]
    L_values = [1,1.2,1.5,2.0,3.0,6.0,12.0,18.0,30.0,60.0,150.0]
    SLO = 25.0
    eps = 0.1

    out = pick_Bopt(batch_sizes, T_values, L_values, SLO, eps)
    print("Bopt:", out['Bopt'], "T:", out['T'], "L:", out['L'])
    #print("Feasible candidates:")
    #for candidate in out['feasible_list']:
    #    print(candidate)