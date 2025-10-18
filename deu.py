import numpy as np
import pandas as pd
import pyomo.environ as pyo

np.random.seed(441)
num_banks = 25
banks = [f"Bank_{i+1:02d}" for i in range(num_banks)]


data = pd.DataFrame({
    "Bank": banks,
    "PersonnelCost": np.random.uniform(80, 850, num_banks),
    "MaterialCost": np.random.uniform(50, 1300, num_banks),
    "Loans": np.random.uniform(1000, 30000, num_banks),
    "Deposits": np.random.uniform(500, 50000, num_banks),
    "Revenues": np.random.uniform(600, 75000, num_banks)
})

inputs = ["PersonnelCost", "MaterialCost"]
outputs = ["Loans", "Deposits", "Revenues"]


def get_input(bank, var):  return float(data.loc[data["Bank"] == bank, var])
def get_output(bank, var): return float(data.loc[data["Bank"] == bank, var])

def solve_dea_for(bank_name, solver="glpk"):
    model = pyo.ConcreteModel()

   
    model.inputs = pyo.Set(initialize=inputs)
    model.outputs = pyo.Set(initialize=outputs)
    model.banks = pyo.Set(initialize=banks)


    model.output_weights = pyo.Var(model.outputs, domain=pyo.NonNegativeReals)
    model.input_weights = pyo.Var(model.inputs, domain=pyo.NonNegativeReals)

    model.efficiency = pyo.Objective(
        expr=sum(model.output_weights[o] * get_output(bank_name, o) for o in model.outputs),
        sense=pyo.maximize
    )

    model.normalization = pyo.Constraint(
        expr=sum(model.input_weights[i] * get_input(bank_name, i) for i in model.inputs) == 1
    )

    def frontier_rule(m, j):
        return sum(m.output_weights[o] * get_output(j, o) for o in m.outputs) - \
               sum(m.input_weights[i] * get_input(j, i) for i in m.inputs) <= 0
    model.frontier = pyo.Constraint(model.banks, rule=frontier_rule)

    solver = pyo.SolverFactory(solver)
    solver.solve(model, tee=False)

    efficiency_score = pyo.value(model.efficiency)
    input_w = {i: pyo.value(model.input_weights[i]) for i in model.inputs}
    output_w = {o: pyo.value(model.output_weights[o]) for o in model.outputs}

    return efficiency_score, input_w, output_w


efficiency = {}
for b in banks:
    score, _, _ = solve_dea_for(b)
    efficiency[b] = score

results = pd.DataFrame({
    "Bank": banks,
    "Efficiency": [efficiency[b] for b in banks]
}).sort_values("Efficiency", ascending=False)

results["Efficient"] = results["Efficiency"].round(6) == 1.0

print("\n--- DEA Efficiency Results (Multiplier Form) ---\n")
print(results)


targets = []
for b in banks:
    eff = efficiency[b]
    if eff < 1:  
        current_inputs = {i: get_input(b, i) for i in inputs}
        reduced_inputs = {i: eff * current_inputs[i] for i in inputs}
        targets.append({
            "Bank": b,
            "Efficiency": eff,
            "PersonnelCost_Current": current_inputs["PersonnelCost"],
            "PersonnelCost_Target": reduced_inputs["PersonnelCost"],
            "MaterialCost_Current": current_inputs["MaterialCost"],
            "MaterialCost_Target": reduced_inputs["MaterialCost"]
        })

targets_df = pd.DataFrame(targets)
print("\n--- Target Reduced Inputs for Inefficient Banks ---\n")
print(targets_df)
