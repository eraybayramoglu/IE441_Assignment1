import random
import pandas as pd
from pyomo.environ import (
    AbstractModel, Set, Param, Var, Objective, Constraint,
    PositiveReals, NonNegativeReals, Binary, minimize, maximize,
    SolverFactory, value
)

SEED = 441
N_DMUS = 25
INPUTS  = ["Cost_Personnel", "Cost_Material"]
OUTPUTS = ["Loans", "Deposits", "Gross_Revenues"]
EPS = 1e-6

def make_data(n_dmus, seed):
    random.seed(seed)
    banks = [f"Bank_{i+1:02d}" for i in range(n_dmus)]
    return pd.DataFrame({
        "Bank": banks,
        "Cost_Personnel":  [random.randint(80, 850)      for _ in banks],
        "Cost_Material":   [random.randint(50, 1300)     for _ in banks],
        "Loans":           [random.randint(1000, 30000) for _ in banks],
        "Deposits":        [random.randint(500, 50000)   for _ in banks],
        "Gross_Revenues":  [random.randint(600, 75000)   for _ in banks],
    })

def to_param(df_like, row_keys, col_keys):
    return {(r, c): float(df_like.loc[df_like["Bank"] == c, r].item())
            for r in row_keys for c in col_keys}

def pack_data(df_like, target_name):
    units = df_like["Bank"].tolist()
    X = to_param(df_like, INPUTS,  units)
    Y = to_param(df_like, OUTPUTS, units)
    T = {u: int(u == target_name) for u in units}
    return {None:{
        "Inputs":  {None: INPUTS},
        "Outputs": {None: OUTPUTS},
        "Units":   {None: units},
        "x": X, "y": Y, "target": T
    }}

def build_ccr():
    m = AbstractModel()
    m.Inputs  = Set(); m.Outputs = Set(); m.Units = Set()
    m.x = Param(m.Inputs,  m.Units, within=PositiveReals)
    m.y = Param(m.Outputs, m.Units, within=PositiveReals)
    m.target = Param(m.Units, within=Binary)
    m.theta = Var(within=NonNegativeReals)
    m.lmbda = Var(m.Units, within=NonNegativeReals)
    def obj(mm): return mm.theta
    m.obj = Objective(rule=obj, sense=minimize)
    def inp(mm, i):
        return sum(mm.lmbda[j]*mm.x[i, j] - mm.theta*mm.x[i, j]*mm.target[j] for j in mm.Units) <= 0
    m.inp_con = Constraint(m.Inputs, rule=inp)
    def out(mm, r):
        return sum(mm.lmbda[j]*mm.y[r, j] - mm.y[r, j]*mm.target[j] for j in mm.Units) >= 0
    m.out_con = Constraint(m.Outputs, rule=out)
    return m

def build_bcc(ccr_model):
    m = ccr_model.clone()
    def convex(mm):
        return sum(mm.lmbda[j] for j in mm.Units) == 1
    m.vrs = Constraint(rule=convex)
    return m

def build_phase2():
    m = AbstractModel()
    m.Inputs  = Set(); m.Outputs = Set(); m.Units = Set()
    m.x = Param(m.Inputs,  m.Units, within=PositiveReals)
    m.y = Param(m.Outputs, m.Units, within=PositiveReals)
    m.target = Param(m.Units, within=Binary)
    m.theta_star = Param(within=PositiveReals)

    m.lmbda = Var(m.Units, within=NonNegativeReals)
    m.s_minus = Var(m.Inputs, within=NonNegativeReals)
    m.s_plus  = Var(m.Outputs, within=NonNegativeReals)

    def obj(mm):
        return sum(mm.s_minus[i] for i in mm.Inputs) + sum(mm.s_plus[r] for r in mm.Outputs)
    m.obj = Objective(rule=obj, sense=maximize)

    def eq_inp(mm, i):
        return mm.s_minus[i] == sum(mm.theta_star*mm.x[i,j]*mm.target[j] - mm.lmbda[j]*mm.x[i,j] for j in mm.Units)
    m.eq_inp = Constraint(m.Inputs, rule=eq_inp)

    def eq_out(mm, r):
        return mm.s_plus[r] == sum(mm.lmbda[j]*mm.y[r,j] - mm.y[r,j]*mm.target[j] for j in mm.Units)
    m.eq_out = Constraint(m.Outputs, rule=eq_out)

    return m

def solver_glpk():
    s = SolverFactory("glpk")
    if not s.available():
        raise RuntimeError("GLPK not available. Install with: conda install -c conda-forge glpk")
    return s

def run_dea(df):
    units = df["Bank"].tolist()
    solver = solver_glpk()
    ccr = build_ccr()
    bcc = build_bcc(ccr)
    ph2 = build_phase2()
    theta_ccr = []
    theta_bcc = []
    lam_ccr = {}
    for k in units:
        data_k = pack_data(df, k)
        m1 = ccr.create_instance(data_k)
        solver.solve(m1)
        t_ccr = float(value(m1.theta))
        theta_ccr.append(t_ccr)
        lam_ccr[k] = {u: float(value(m1.lmbda[u])) for u in units}
        m2 = bcc.create_instance(data_k)
        solver.solve(m2)
        theta_bcc.append(float(value(m2.theta)))
    rts = {}
    for k in units:
        s_lambda = sum(lam_ccr[k].values())
        if abs(s_lambda - 1.0) <= EPS:
            rts[k] = "CRS"
        elif s_lambda > 1.0 + EPS:
            rts[k] = "DRS"
        else:
            rts[k] = "IRS"
    ccr_rank = pd.DataFrame({
        "Bank": units,
        "Theta_CCR": theta_ccr,
        "RTS": [rts[k] for k in units]
    }).sort_values("Theta_CCR", ascending=False).reset_index(drop=True)
    bcc_rank = pd.DataFrame({
        "Bank": units,
        "Theta_BCC": theta_bcc
    }).sort_values("Theta_BCC", ascending=False).reset_index(drop=True)

    sminus_all = {}
    splus_all = {}
    slack_sum = {}
    for k, t in zip(units, theta_ccr):
        data_k = pack_data(df, k)
        data_k[None]["theta_star"] = {None: t}
        m_ph2 = ph2.create_instance(data_k)
        solver.solve(m_ph2)
        sminus_all[k] = {i: float(value(m_ph2.s_minus[i])) for i in INPUTS}
        splus_all[k]  = {r: float(value(m_ph2.s_plus[r]))  for r in OUTPUTS}
        slack_sum[k] = sum(sminus_all[k].values()) + sum(splus_all[k].values())

    rows = []
    for k, t in zip(units, theta_ccr):
        if t < 1 - EPS:
            row = df.loc[df.Bank == k].iloc[0]
            x1 = float(row[INPUTS[0]]); x2 = float(row[INPUTS[1]])
            s1 = sminus_all[k][INPUTS[0]]; s2 = sminus_all[k][INPUTS[1]]
            y0 = float(row[OUTPUTS[0]]); y1 = float(row[OUTPUTS[1]]); y2 = float(row[OUTPUTS[2]])
            sp0 = splus_all[k][OUTPUTS[0]]; sp1 = splus_all[k][OUTPUTS[1]]; sp2 = splus_all[k][OUTPUTS[2]]

            x1_proj = t*x1 - s1
            x2_proj = t*x2 - s2
            y0_proj = y0 + sp0
            y1_proj = y1 + sp1
            y2_proj = y2 + sp2

            peers = [u for u in units if lam_ccr[k][u] > 1e-6]
            rows.append({
                "Bank": k,
                "Theta_CCR": t,
                "SlackSum": slack_sum[k],
                f"Target_{INPUTS[0]}":  x1_proj,
                f"Target_{INPUTS[1]}":  x2_proj,
                f"Target_{OUTPUTS[0]}": y0_proj,
                f"Target_{OUTPUTS[1]}": y1_proj,
                f"Target_{OUTPUTS[2]}": y2_proj,
                "Peers": ", ".join(peers)
            })
    targets = pd.DataFrame(rows)
    return ccr_rank, bcc_rank, targets, theta_ccr, theta_bcc, lam_ccr, slack_sum

df = make_data(N_DMUS, SEED)

ccr_rank, bcc_rank, targets, theta_ccr, theta_bcc, lam_ccr, slack_sum = run_dea(df)

print("CCR (input-oriented)")
print(ccr_rank)
print()

print("BCC (VRS)")
print(bcc_rank)
print()

if not targets.empty:
    print("Radial targets (CCR-inefficient)")
    print(targets)
    print()
else:
    print("All banks are CCR-efficient.")
    print()


# Q5 / Q7 with safe bank-\u03b8 mapping (avoid order issues after sorting)
units = df["Bank"].tolist()
theta_ccr_map = {u: t for u, t in zip(units, theta_ccr)}
theta_bcc_map = {u: t for u, t in zip(units, theta_bcc)}

# Q5: Radial vs CCR-efficient
radial_eff = set([u for u in units if abs(theta_ccr_map[u] - 1.0) <= EPS])
ccr_eff    = set([u for u in units if abs(theta_ccr_map[u] - 1.0) <= EPS and abs(slack_sum.get(u, 0.0)) <= EPS])
print("Q5: Radial vs CCR-efficient")
print("Radially efficient:", sorted(radial_eff))
print("CCR-efficient:",    sorted(ccr_eff))
print("Lose efficiency when requiring zero slack (radial - CCR):", sorted(radial_eff - ccr_eff))
print()

# Q7: CCR vs BCC
bcc_eff = set([u for u in units if abs(theta_bcc_map[u] - 1.0) <= EPS])
print("Q7: CCR vs BCC")
print("CCR-efficient:",               sorted(ccr_eff))
print("BCC-efficient:",               sorted(bcc_eff))
print("Newly efficient under BCC:",   sorted(bcc_eff - ccr_eff))
print("Efficient in CCR but not BCC:", sorted(ccr_eff - bcc_eff))
print()

# Q8: Merge Test (BCC)

def merge_worst_two_and_test_bcc(df_local, theta_bcc_list):
    units_local = df_local["Bank"].tolist()
    order = sorted(range(len(units_local)), key=lambda i: theta_bcc_list[i])
    a, b = units_local[order[0]], units_local[order[1]]

    merged = pd.DataFrame([{
        "Bank": f"Merged_{a}_{b}",
        "Cost_Personnel": float(df_local.loc[df_local.Bank==a,"Cost_Personnel"].item() + df_local.loc[df_local.Bank==b,"Cost_Personnel"].item()),
        "Cost_Material":  float(df_local.loc[df_local.Bank==a,"Cost_Material"].item()  + df_local.loc[df_local.Bank==b,"Cost_Material"].item()),
        "Loans":          float(df_local.loc[df_local.Bank==a,"Loans"].item()          + df_local.loc[df_local.Bank==b,"Loans"].item()),
        "Deposits":       float(df_local.loc[df_local.Bank==a,"Deposits"].item()       + df_local.loc[df_local.Bank==b,"Deposits"].item()),
        "Gross_Revenues": float(df_local.loc[df_local.Bank==a,"Gross_Revenues"].item() + df_local.loc[df_local.Bank==b,"Gross_Revenues"].item()),
    }])
    df_m = pd.concat([df_local, merged], ignore_index=True)
    units_m = df_m["Bank"].tolist()

    bcc_m = build_bcc(build_ccr())
    data_m = {None:{
        "Inputs":  {None: INPUTS},
        "Outputs": {None: OUTPUTS},
        "Units":   {None: units_m},
        "x": to_param(df_m, INPUTS,  units_m),
        "y": to_param(df_m, OUTPUTS, units_m),
        "target": {u: int(u == merged.iloc[0]["Bank"]) for u in units_m}
    }}
    s = solver_glpk()
    m = bcc_m.create_instance(data_m)
    s.solve(m)
    theta_m = float(value(m.theta))
    peers_m = [u for u in m.Units if value(m.lmbda[u]) > 1e-6]
    return f"{a} & {b}", theta_m, peers_m

pair, theta_m, peers_m = merge_worst_two_and_test_bcc(df, theta_bcc)
print("Q8: Merge Test (BCC)")
print("Merged pair:", pair)
print(f"Merged bank BCC theta: {theta_m:.4f}")
print("Peers (merged):", peers_m)