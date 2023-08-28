# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 3:42 PM
# @Author  : Shijin Yang
# @FileName: main.py

from heuristics import Heuristic
from MIP import MIP
import pandas as pd

if __name__ == "__main__":
    file_list = list(range(1, 9))
    runway_list = list(range(1, 5))
    # MIP
    al = MIP()
    result = []
    for f in file_list:
        fname = f'airland{f}.txt'
        print("filename:", fname)
        for r in runway_list:
            print("runway: ", r)
            _, obj, gap, s_run_time, s_arrived_percent, s_stdev = al.static_schedule(fname, number_runways=r)
            _, Z_sol, Z_disp, d_run_time, d_arrived_percent, d_stdev, D_max = al.dynamic_schedule(fname,
                                                                                                  number_runways=r,)
            rec = [f, r, obj, gap, s_run_time, s_arrived_percent, s_stdev,
                   Z_sol, Z_disp, D_max, d_run_time, d_arrived_percent, d_stdev]
            result.append(rec)
            print("---------------------------")
    col_names = ["Problem Number", "Number of runways", "Static optimal solution", "Static gap", "Static run time",
                 "Static arrived at target ratio", "Static standard deviation",
                 "DALP-OPT Z_sol", "DALP-OPT Z_disp", "DALP-OPT D_max", "DALP-OPT run time",
                 "DALP-OPT arrived at target ratio", "DALP-OPT standard deviation"]
    df = pd.DataFrame(result, columns=col_names)
    df.to_excel(f"results/MIP-result({file_list[0]}-{file_list[-1]}).xlsx", index=False)

    print("=====================================================")

    # Heuristic
    print("Heuristic")
    al = Heuristic()
    result = []
    for f in file_list:
        file_name = f'airland{f}.txt'
        print("filename:", file_name)
        for r in runway_list:
            print("runway: ", r)
            obj, s_run_time, s_arrived_percent, s_stdev = al.static_heuristic(filename=file_name,
                                                                              number_runways=r)
            Z_sol, Z_disp, D_max, d_run_time, d_arrived_percent, d_stdev = al.dynamic_heuristic(filename=file_name,
                                                                                                number_runways=r)
            rec = [f, r, obj, s_run_time, s_arrived_percent, s_stdev,
                   Z_sol, Z_disp, D_max, d_run_time, d_arrived_percent, d_stdev]
            result.append(rec)
            print("---------------------------")
    col_names = ["Problem Number", "Number of runways", "Static solution", "Static run time",
                 "Static arrived at target ratio", "Static standard deviation", "DALP-H1 Z_sol", "DALP-H1 Z_disp",
                 "DALP-H1 D_max", "DALP-H1 run time", "DALP-H1 arrived at target ratio", "DALP-H1 standard deviation"]
    df = pd.DataFrame(result, columns=col_names)
    df.to_excel(f"results/Heuristic-result({file_list[0]}-{file_list[-1]}).xlsx", index=False)
