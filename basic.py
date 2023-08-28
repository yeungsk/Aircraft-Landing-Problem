# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 4:19 PM
# @Author  : Shijin Yang
# @FileName: basic.py

import os
import numpy as np


class AircraftLanding(object):
    def __init__(self):
        self.number_aircraft: int = 0
        self.freeze_time: int = 0
        self.aircraft: list = []
        self.separation_time: list = []
        self.target_time: dict = {}
        self.earliest_time: dict = {}
        self.latest_time: dict = {}
        self.penalty_cost_before: dict = {}
        self.penalty_cost_after: dict = {}

    def set_data(self, filename):
        data = open(os.getcwd() + "/data/" + filename, 'r')
        lines = data.readlines()
        number_aircraft = int(lines[0].split()[0])
        freeze_time = int(lines[0].split()[1])

        # index 0: appearance time
        # index 1: earliest landing time
        # index 2: target landing time
        # index 3: latest landing time
        # index 4: penalty cost per unit of time for landing before target
        # index 5: penalty cost per unit of time for landing after target
        aircraft = np.empty([number_aircraft, 6], dtype=float)
        # separation time
        separation_time = np.empty([number_aircraft, number_aircraft], dtype=int)

        # put all the data to a single list
        data_list = []
        for line in lines[1:]:
            data_list = data_list + line.split()

        index = 0
        for x in range(0, len(data_list), number_aircraft + 6):
            aircraft_x = data_list[x:x + 6 + number_aircraft]
            # the first 6 are the details of the aircraft
            aircraft[index] = [float(i) for i in aircraft_x[:6]]
            # the remains are the separation time
            separation_time[index] = [int(i) for i in aircraft_x[6:]]
            index += 1
        data.close()

        self.number_aircraft = number_aircraft
        self.freeze_time = freeze_time
        self.aircraft = aircraft
        self.separation_time = separation_time
        self.earliest_time = {i: self.aircraft[i, 1] for i in range(number_aircraft)}
        self.target_time = {i: self.aircraft[i, 2] for i in range(number_aircraft)}
        self.latest_time = {i: self.aircraft[i, 3] for i in range(number_aircraft)}
        self.penalty_cost_before = {i: self.aircraft[i, 4] for i in range(number_aircraft)}
        self.penalty_cost_after = {i: self.aircraft[i, 5] for i in range(number_aircraft)}
