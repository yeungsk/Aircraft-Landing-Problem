# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 5:34 PM
# @Author  : Shijin Yang
# @FileName: heuristic.py

import time as tm
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from basic import AircraftLanding


class Heuristic(AircraftLanding):
    def __init__(self):
        super().__init__()

    def set_data(self, filename):
        super().set_data(filename)

    def ALPF(self, stype: str, aircraft_list: list, penalty_cost_function: int = 0, frozen: dict = None, X: dict = {},
             lambda_cost: int = 1, lambda_disp: int = 1, lambda_max: int = 0, p: int = 1,
             max_displacement: float = float("inf"),):
        model = gp.Model("LP")
        model.Params.LogToConsole = 0

        E = {k: v for k, v in self.earliest_time.items() if k in aircraft_list}
        T = {k: v for k, v in self.target_time.items() if k in aircraft_list}
        L = {k: v for k, v in self.latest_time.items() if k in aircraft_list}
        g = {k: v for k, v in self.penalty_cost_before.items() if k in aircraft_list}
        h = {k: v for k, v in self.penalty_cost_after.items() if k in aircraft_list}
        S = {(i, j): self.separation_time[i, j] for j in aircraft_list for i in aircraft_list}

        # Decision variables
        x = model.addVars(aircraft_list, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
        # how soon plane i lands before Ti
        alpha = model.addVars(g.keys(), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha")
        # how soon plan i lands after T
        beta = model.addVars(h.keys(), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta")

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
            disp_cost = p * gp.quicksum(D[i] for i in list(X.keys()))
            model.setObjective(lambda_cost * original_cost + lambda_disp * disp_cost + lambda_max * D_max, GRB.MINIMIZE)
        else:
            # static case
            if penalty_cost_function != 0:
                model.setObjective(gp.quicksum(g[i] * f_g[i] + h[i] * f_h[i] for i in aircraft_list), GRB.MINIMIZE)
            else:
                model.setObjective(gp.quicksum(g[i] * alpha[i] + h[i] * beta[i] for i in aircraft_list), GRB.MINIMIZE)

        # Constraints
        # separation constraint
        model.addConstrs(
            (x[aircraft_list[i + 1]] - x[aircraft_list[i]] >= S[(aircraft_list[i], aircraft_list[i + 1])] for i in
             range(0, len(aircraft_list) - 1)), name="c1")

        # Alpha: At least as big as zero and the time difference between T_t and x_i
        # and at most the time difference between T_t and E_t
        model.addConstrs((alpha[i] >= T[i] - x[i] for i in aircraft_list), name="c2")
        model.addConstrs((alpha[i] <= T[i] - E[i] for i in aircraft_list), name="c3")

        # Similar equations for beta
        model.addConstrs((beta[i] >= x[i] - T[i] for i in aircraft_list), name="c4")
        model.addConstrs((beta[i] <= L[i] - T[i] for i in aircraft_list), name="c5")

        # The landing time (x_t) to the time i lands before (alpha_i) or after (beta_i),target(T_i).
        model.addConstrs((x[i] == T[i] - alpha[i] + beta[i] for i in aircraft_list), name="c6")

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
            # If beta > 10, f_h = 10 * alpha
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
            # If beta > 10, f_h = 5 * alpha
            for i in aircraft_list:
                model.addGenConstrPWL(alpha[i], f_g[i], [0, 5, 5, 10, 10, 10000],
                                      [0, 2 * 5, 4 * 5, 4 * 10, 10 * 10, 10 * 10000], name="Piecewise alpha")
                model.addGenConstrPWL(beta[i], f_h[i], [0, 5, 5, 10, 10, 10000],
                                      [0, 1 * 5, 2 * 5, 2 * 10, 5 * 10, 5 * 10000], name="Piecewise beta")

        model.optimize()

        schedule_times = {}
        if stype == "dynamic":
            for i in aircraft_list:
                schedule_times[i] = model.getVarByName("x[" + str(i) + "]").X
            D_max = model.getVarByName("D_max").X
            return schedule_times, original_cost.getValue(), disp_cost.getValue(), D_max
        else:
            ov = round(model.objVal)
            for i in aircraft_list:
                schedule_times[i] = model.getVarByName("x[" + str(i) + "]").X
            return ov, schedule_times

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

    def static_heuristic(self, filename: str, number_runways: int = 1, penalty_cost_function: int = 0):
        self.set_data(filename)
        aircraft_list = list(range(0, self.number_aircraft))
        T = {k: v for k, v in self.target_time.items() if k in aircraft_list}  # target landing time
        S = {(i, j): self.separation_time[i, j] for j in aircraft_list for i in aircraft_list}  # separation time

        time_start = tm.process_time()
        ascending_T_index = [i[0] for i in sorted(T.items(), key=lambda item: item[1])]
        A = {r_index: [] for r_index in range(0, number_runways)}
        x = {}
        for j in ascending_T_index:
            B_jr = {}
            for r in range(0, number_runways):
                B_jr[r] = []
                mS = []
                if A[r]:
                    for k in A[r]:
                        mS.append(x[k] + S[(k, j)])
                if mS:
                    B_jr[r] = max(T[j], max(mS))
                else:
                    B_jr[r] = T[j]
            # the best landing time for plane j
            x[j] = min(B_jr.items(), key=lambda item: item[1])[1]
            # feasible runway for x(j)
            rway = min(B_jr.items(), key=lambda item: item[1])[0]
            # put j into the runway r
            A[rway].append(j)

        cost = None
        schedule_times = {}
        # recalculate the landing time
        for r, ids in A.items():
            if ids:
                ov, stimes = self.ALPF(stype="static", aircraft_list=ids, penalty_cost_function=penalty_cost_function)
                if cost is None:
                    cost = ov
                else:
                    cost += ov
                schedule_times = {**schedule_times, **stimes}

        time_end = tm.process_time()
        total_time = round((time_end - time_start), 3)
        print("Static: ")
        print("total cost: ", cost)
        print("Total time (s): %s" % total_time)
        arrived_at_target_percent, standard_deviation = self.percent_stdev(schedule_times)
        print("------")
        return cost, total_time, arrived_at_target_percent, standard_deviation

    def dynamic_heuristic(self, filename: str, number_runways: int = 1, penalty_cost_function: int = 0,
                          lambda_cost: int = 1, lambda_disp: int = 1, lambda_max: int = 0, p: int = 1,
                          max_displacement: float = float("inf")):
        self.set_data(filename)
        aircraft_list = list(range(0, self.number_aircraft))
        T = {k: v for k, v in self.target_time.items() if k in aircraft_list}
        ascending_T_index = [i[0] for i in sorted(T.items(), key=lambda item: item[1])]
        S = {(i, j): self.separation_time[i, j] for j in aircraft_list for i in aircraft_list}  # separation time
        time_start = tm.process_time()
        X = {i: float("inf") for i in aircraft_list}
        # any aircraft assigned a landing time within t_star of the current time had its landing time frozen
        t_star = self.freeze_time

        # Initialization
        F0 = ascending_T_index.copy()  # the set of aircraft that have not yet appeared by time t
        F1 = []  # aircraft that have appeared by time t, but have not yet landed or had their landing times frozen
        F2 = []  # aircraft that have appeared by time t and have landed or had their landing times frozen
        Z_disp = 0  # accumulated displacement cost
        Total_D_max = 0
        # get minimum target time
        t, index = self.aircraft[F0[0], 0], F0[0]
        F1.append(index)
        F0.remove(index)
        X[index] = self.target_time[index]
        Z_sol = 0
        iteration = 0
        while len(F0) > 0:
            iteration += 1
            # Get next's plane and it's index
            t, index = self.aircraft[F0[0], 0], F0[0]
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
            before_sort = F1 + F2
            ids = [x for _, x in sorted(zip([ascending_T_index.index(x) for x in before_sort], before_sort))]
            # Heuristic Part
            A = {r_index: [] for r_index in range(0, number_runways)}
            x = {}
            for j in ids:
                B_jr = {}
                for r in range(0, number_runways):
                    B_jr[r] = []
                    mS = []
                    if A[r]:
                        for k in A[r]:
                            mS.append(x[k] + S[(k, j)])
                    if mS:
                        B_jr[r] = max(T[j], max(mS))
                    else:
                        B_jr[r] = T[j]
                # the best landing time for plane j
                x[j] = min(B_jr.items(), key=lambda item: item[1])[1]
                # feasible runway for x(j)
                rway = min(B_jr.items(), key=lambda item: item[1])[0]
                # put j into the runway r
                A[rway].append(j)
            # recalculate the landing time
            x_new = {}
            Z_sol_i = None
            Z_disp_i = None
            for r, ids in A.items():
                if ids:
                    aircraft_with_time_frozen = {i: j for i, j in X.items() if i in F2 and i in ids}
                    x_new_i, ori_i, disp_i, D_max = self.ALPF(stype="dynamic", aircraft_list=ids,
                                                              frozen=aircraft_with_time_frozen,
                                                              X={i: j for i, j in X.items()
                                                                 if i in [item for item in F1
                                                                          if item != index and item in ids]},
                                                              lambda_cost=lambda_cost,
                                                              lambda_disp=lambda_disp,
                                                              lambda_max=lambda_max,
                                                              p=p,
                                                              max_displacement=max_displacement,
                                                              penalty_cost_function=penalty_cost_function)
                    if Z_sol_i is not None:
                        Z_sol_i += ori_i
                    else:
                        Z_sol_i = ori_i
                    if Z_disp_i is not None:
                        Z_disp_i += disp_i
                    else:
                        Z_disp_i = disp_i
                    if round(D_max) >= Total_D_max:
                        Total_D_max = D_max

                    x_new = {**x_new_i, **x_new}

            if Z_sol_i is not None:
                Z_sol = Z_sol_i
            Z_disp += Z_disp_i
            for ind, time in x_new.items():
                X[ind] = time

        print("Dynamic:")
        time_end = tm.process_time()
        total_time = round((time_end - time_start), 3)
        print("Original cost: ", Z_sol)
        print("accumulated displacement cost : ", Z_disp)
        print("D max: ", Total_D_max)
        print("Total time (s): %s" % total_time)
        arrived_at_target_percent, standard_deviation = self.percent_stdev(X)
        return round(Z_sol, 2), round(Z_disp, 2), Total_D_max, total_time, arrived_at_target_percent, standard_deviation


if __name__ == "__main__":
    al = Heuristic()
    al.static_heuristic(filename="airland2.txt", number_runways=1)
    al.dynamic_heuristic(filename="airland1.txt", number_runways=1)
