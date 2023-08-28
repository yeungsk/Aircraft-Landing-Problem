# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 2:45 PM
# @Author  : Shijin Yang
# @FileName: dynamic_case.py

import time as tm
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from basic import AircraftLanding


class MIP(AircraftLanding):
    def __init__(self):
        super().__init__()

    def set_data(self, filename):
        super().set_data(filename)

    def ALP(self, aircraft_list: list, number_runways: int, stype: str, penalty_cost_function: int = 0,
            frozen: dict = None, X: dict = {}, lambda_cost: int = 1, lambda_disp: int = 1,
            lambda_max: int = 0, p: int = 1, max_displacement: float = float("inf")):
        # create a Gurobi model
        model = gp.Model("aircraft scheduling")
        model.Params.LogToConsole = 0
        model.Params.TimeLimit = 600

        E = {k: v for k, v in self.earliest_time.items() if k in aircraft_list}
        T = {k: v for k, v in self.target_time.items() if k in aircraft_list}
        L = {k: v for k, v in self.latest_time.items() if k in aircraft_list}
        g = {k: v for k, v in self.penalty_cost_before.items() if k in aircraft_list}
        h = {k: v for k, v in self.penalty_cost_after.items() if k in aircraft_list}
        ij_list = [(i, j) for j in aircraft_list for i in aircraft_list]
        ir_list = [(i, r) for r in range(number_runways) for i in aircraft_list]
        s_ub = {(i, j): self.separation_time[i, j] for j in aircraft_list for i in aircraft_list}

        # Decision variables
        x = model.addVars(aircraft_list, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
        # how soon plane i lands before Ti
        alpha = model.addVars(g.keys(), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha")
        # how soon plan i lands after T
        beta = model.addVars(h.keys(), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta")
        # if plane i lands before plane j
        d = model.addVars(ij_list, lb=0, ub=1, vtype=GRB.BINARY, name="d")
        # z_ij whether planes i and j land on the same runway
        z = model.addVars(ij_list, lb=0, ub=1, vtype=GRB.BINARY, name="z")
        # y_ir whether plane i lands on runway r
        y = model.addVars(ir_list, lb=0, ub=1, vtype=GRB.BINARY, name="y")
        # the required separation time between plane i landing and plane j landing (where planes i lands before plane j
        # and they land on different runways
        s = model.addVars(ij_list, lb=0, ub=s_ub, vtype=GRB.CONTINUOUS, name="s")

        # linear or piecewise linear
        if penalty_cost_function != 0:
            f_g = model.addVars(aircraft_list, name="f_g")
            f_h = model.addVars(aircraft_list, name="f_h")

        if stype == "dynamic":
            D_dict = {i: 0 for i in list(X.keys())}
            # the contribution to total displacement
            D = model.addVars(D_dict.keys(), lb=0, ub=GRB.INFINITY, obj=D_dict, vtype=GRB.CONTINUOUS, name="D")
            D_max = model.addVar(lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="D_max")
            # X[i] - x[i]
            X_x = model.addVars(list(X.keys()), lb=0, obj=0, vtype=GRB.CONTINUOUS, name="X_x")
            # x[i] - X[i]
            x_X = model.addVars(list(X.keys()), lb=0, obj=0, vtype=GRB.CONTINUOUS, name="x_X")

            # Objective
            # the cost of the solution in terms of the cost function associated with the original (static) ALP
            if penalty_cost_function != 0:
                original_cost = gp.quicksum(g[i] * f_g[i] + h[i] * f_h[i] for i in aircraft_list)
            else:
                original_cost = gp.quicksum(g[i] * alpha[i] + h[i] * beta[i] for i in aircraft_list)
            # the displacement cost component
            disp_cost = (p * gp.quicksum(D[i] for i in list(X.keys())))
            model.setObjective(lambda_cost * original_cost + lambda_disp * disp_cost + lambda_max * D_max, GRB.MINIMIZE)

        else:
            # static case
            if penalty_cost_function != 0:
                model.setObjective(gp.quicksum(g[i] * f_g[i] + h[i] * f_h[i] for i in aircraft_list), GRB.MINIMIZE)
            else:
                model.setObjective(gp.quicksum(g[i] * alpha[i] + h[i] * beta[i] for i in aircraft_list), GRB.MINIMIZE)

        # Constraints
        # Each plane lands within its time window
        model.addConstrs((x[i] >= E[i] for i in aircraft_list), name="c1_1")
        model.addConstrs((x[i] <= L[i] for i in aircraft_list), name="c1_2")

        # Either plane / must land before plane j or plane j must land before plane i
        model.addConstrs((d[i, j] + d[j, i] == 1 for j in aircraft_list for i in aircraft_list if j != i), name="c2")

        # Plane i must land before j for those pairs in set W and V
        model.addConstrs((d[i, j] == 1 for j in aircraft_list for i in aircraft_list if L[i] < E[j]), name="c3")

        # Alpha: At least as big as zero and the time difference between T_t and x_i
        # and at most the time difference between T_t and E_t
        model.addConstrs((alpha[i] >= T[i] - x[i] for i in aircraft_list), name="c4")
        model.addConstrs((alpha[i] <= T[i] - E[i] for i in aircraft_list), name="c5")

        # Similar equations for beta
        model.addConstrs((beta[i] >= x[i] - T[i] for i in aircraft_list), name="c6")
        model.addConstrs((beta[i] <= L[i] - T[i] for i in aircraft_list), name="c7")

        # The landing time (x_t) to the time i lands before (alpha_i) or after (beta_i),target(T_i).
        model.addConstrs((x[i] == T[i] - alpha[i] + beta[i] for i in aircraft_list), name="c8")

        # Each planes lands on exactly one runway
        model.addConstrs((gp.quicksum(y[i, r] for r in range(number_runways)) == 1 for i in aircraft_list), name="c9")

        # If i and j land on the same runway so do j and i
        model.addConstrs((z[i, j] == z[j, i] for i in aircraft_list for j in aircraft_list if i != j), name="c10")

        # If there is any runway r for which y_ir and y_jr are both one, then we force z_ij to be one.
        # If z_ij = 0 then planes i and j cannot land on the same runway
        model.addConstrs((z[i, j] >= y[i, r] + y[j, r] - 1 for r in range(number_runways)
                          for j in aircraft_list for i in aircraft_list if i != j), name="c11")

        # Separation constraint for pairs of planes in V
        model.addConstrs((x[j] >= x[i] + self.separation_time[i, j] * z[i, j] + s[i, j] * (1 - z[i, j])
                          for j in aircraft_list for i in aircraft_list
                          if (i != j) and (L[i] < E[j]) and (L[i] + self.separation_time[i, j] > E[j])), name="c12")

        # Separation constraint for pairs of planes in U
        model.addConstrs((x[j] >= x[i] + self.separation_time[i, j] * z[i, j] + s[i, j] * (1 - z[i, j]) - (
                L[i] + self.separation_time[i, j] - E[j]) * d[j, i]
                          for j in aircraft_list for i in aircraft_list
                          if i != j and (E[j] <= E[i] <= L[j] or E[j] <= L[i] <= L[j] or E[i] <= E[j] <= L[i] or
                                         E[i] <= L[j] <= L[i])),
                         name="c13")

        # Dynamic constraints
        if stype == "dynamic":
            # if some landing times are frozen, they can't be changed
            if frozen is not None:
                model.addConstrs((x[i] == j for i, j in frozen.items()), name="landing times frozen")
            # displacement constraints
            model.addConstrs((X_x[i] >= X[i] - x[i] for i in list(X.keys())), name="X_i minus x_i")
            model.addConstrs((x_X[i] >= x[i] - X[i] for i in list(X.keys())), name="x_i minus X_i")
            model.addConstrs((D[i] >= g[i] * X_x[i] for i in list(X.keys()) if X[i] < T[i]),
                             name="displacement if originally scheduled to land before target time")
            model.addConstrs((D[i] >= h[i] * x_X[i] for i in list(X.keys()) if X[i] > T[i]),
                             name="displacement if originally scheduled to land after target time")
            model.addConstrs((D[i] >= g[i] * X_x[i] + h[i] * x_X[i] for i in list(X.keys()) if X[i] == T[i]),
                             name="displacement if originally scheduled to land at target time")
            model.addConstrs((D[i] <= max_displacement for i in list(X.keys())), name="Limiting displacement")
            model.addGenConstrMax(D_max, D, 0, name="max displacement")

        # Piecewise constraints
        # f_g = alpha * the penalty cost coefficient
        # f_h = beta * the penalty cost coefficient
        if penalty_cost_function == 1:
            # If alpha <= 5, f_g = 1 * alpha
            # If 5 < alpha <= 10, f_g = 2 * alpha
            # If alpha > 10, f_g = 5 * alpha
            # So does beta
            for i in aircraft_list:
                model.addGenConstrPWL(alpha[i], f_g[i], [0, 5, 5, 10, 10, 10000],
                                      [0, 1 * 5, 2 * 5, 2 * 10, 5 * 10, 5 * 10000], name="Piecewise alpha")
                model.addGenConstrPWL(beta[i], f_h[i], [0, 5, 5, 10, 10, 10000],
                                      [0, 1 * 5, 2 * 5, 2 * 10, 5 * 10, 5 * 10000], name="Piecewise beta")
        elif penalty_cost_function == 2:
            # If alpha <= 5, f_g = 1 * alpha
            # If 5 < alpha <= 10, f_g = 2 * alpha
            # If alpha > 10, f_g = 5 * alpha
            # If beta <= 5, f_h = 2 * beta
            # If 5 < beta <= 10, f_h = 4 * beta
            # If beta > 10, f_h = 10 * beta
            for i in aircraft_list:
                model.addGenConstrPWL(alpha[i], f_g[i], [0, 5, 5, 10, 10, 10000],
                                      [0, 1 * 5, 2 * 5, 2 * 10, 5 * 10, 5 * 10000], name="Piecewise alpha")
                model.addGenConstrPWL(beta[i], f_h[i], [0, 5, 5, 10, 10, 10000],
                                      [0, 2 * 5, 4 * 5, 4 * 10, 10 * 10, 10 * 10000], name="Piecewise beta")
        elif penalty_cost_function == 3:
            # If alpha <= 5, f_g = 2 * alpha
            # If 5 < alpha <= 10, f_g = 4 * alpha
            # If alpha > 10, f_g = 10 * alpha
            # If beta <= 5, f_h = 1 * beta
            # If 5 < beta <= 10, f_h = 2 * beta
            # If beta > 10, f_h = 5 * beta
            for i in aircraft_list:
                model.addGenConstrPWL(alpha[i], f_g[i], [0, 5, 5, 10, 10, 10000],
                                      [0, 2 * 5, 4 * 5, 4 * 10, 10 * 10, 10 * 10000], name="Piecewise alpha")
                model.addGenConstrPWL(beta[i], f_h[i], [0, 5, 5, 10, 10, 10000],
                                      [0, 1 * 5, 2 * 5, 2 * 10, 5 * 10, 5 * 10000], name="Piecewise beta")

        model.optimize()

        if stype == "static":
            schedule_times = {}
            for i in aircraft_list:
                schedule_times[i] = model.getVarByName("x[" + str(i) + "]").X
            ov = round(model.objVal)
            gap = model.MIPGap
            rt = model.Runtime
            print(f"Objective value: {ov}\nSolver runtime (s): {rt}\nMIP gap: {gap}\n")
            return schedule_times, ov, gap, round(rt, 3)

        elif stype == "dynamic":
            schedule_times = {}
            for i in aircraft_list:
                schedule_times[i] = model.getVarByName("x[" + str(i) + "]").X
            D_max = model.getVarByName("D_max").X
            return schedule_times, original_cost.getValue(), disp_cost.getValue(), D_max

    def percent_stdev(self, schedule_times):
        difference_abs = {i: abs(self.target_time[i] - round(schedule_times[i])) for i in range(self.number_aircraft)}
        # calculate the percent of aircraft that land on the target time
        arrived_at_target = 0
        for k, v in difference_abs.items():
            if v <= 0:
                arrived_at_target += 1
        arrived_at_target_percent = round((arrived_at_target / self.number_aircraft), 4)
        # calculate the standard deviation
        standard_deviation = round(np.std(list(difference_abs.values())), 2)
        return arrived_at_target_percent, standard_deviation

    def static_schedule(self, filename, number_runways, penalty_cost_function: int = 0):
        self.set_data(filename)  # set data
        schedule_times, obj_value, gap, run_time = self.ALP(aircraft_list=list(range(self.number_aircraft)),
                                                            number_runways=number_runways, stype="static",
                                                            penalty_cost_function=penalty_cost_function)
        arrived_at_target_percent, standard_deviation = self.percent_stdev(schedule_times)

        return schedule_times, round(obj_value), gap, run_time, arrived_at_target_percent, standard_deviation

    def dynamic_schedule(self, filename, number_runways: int, lambda_cost: int = 1, lambda_disp: int = 1,
                         lambda_max: int = 0, p: int = 1, max_displacement: float = float("inf"),
                         penalty_cost_function: int = 0):
        self.set_data(filename)  # set data

        X = {i: float("inf") for i in range(self.number_aircraft)}  # landing time
        # any aircraft assigned a landing time within t_star of the current time had its landing time frozen
        t_star = self.freeze_time
        time_start = tm.process_time()
        # Initialization
        F0 = [i for i in range(self.number_aircraft)]  # the set of aircraft that have not yet appeared by time t
        F1 = []  # aircraft that have appeared by time t, but have not yet landed or had their landing times frozen
        F2 = []  # aircraft that have appeared by time t and havel or had their landing times frozen
        Z_disp = 0  # accumulated displacement cost
        Total_D_max = 0
        t, index = (min((self.aircraft[i, 0], i) for i in F0))
        F1.append(index)
        F0.remove(index)
        X[index] = self.target_time[index]
        while len(F0) > 0:
            # Get next's plane and it's index
            t, index = (min((self.aircraft[i, 0], i) for i in F0))
            # Calculate F0, F1, F2
            F1.append(index)
            F0.remove(index)
            # Check if any planes in F1 have their landing time frozen by current time t
            frozen = []
            for i in F1:
                if X[i] <= t + t_star:
                    frozen.append(i)
            F1 = list(set(F1) - set(frozen))
            F2 = list(set(F2) | set(frozen))
            aircraft_with_time_frozen = {i: j for i, j in X.items() if i in F2}
            x_new, Z_sol, disp, D_max = self.ALP(F1 + F2, number_runways, "dynamic", frozen=aircraft_with_time_frozen,
                                                 X={i: j for i, j in X.items()
                                                    if i in [item for item in F1 if item != index]},
                                                 lambda_cost=lambda_cost, lambda_disp=lambda_disp,
                                                 lambda_max=lambda_max, p=p,
                                                 max_displacement=max_displacement,
                                                 penalty_cost_function=penalty_cost_function)
            Z_disp += disp
            for ind, time in x_new.items():
                X[ind] = time
            if round(D_max) >= Total_D_max:
                Total_D_max = round(D_max)

        time_end = tm.process_time()
        total_time = round(time_end - time_start, 3)
        print("\nDynamic")
        print("Total Landing Cost: %s" % round(Z_sol, 2))
        print("The accumulated displacement cost: %s" % round(Z_disp, 2))
        print("Total time (s): %s" % total_time)
        print("D_max: %s" % Total_D_max)
        arrived_at_target_percent, standard_deviation = self.percent_stdev(X)
        return X, round(Z_sol, 2), round(Z_disp, 2), total_time, arrived_at_target_percent, standard_deviation, Total_D_max


if __name__ == "__main__":
    al = MIP()
    al.static_schedule("airland1.txt", number_runways=1)
    al.dynamic_schedule("airland1.txt", number_runways=1)
