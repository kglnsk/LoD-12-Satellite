#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import os
from tqdm.auto import trange,tqdm
import pandas as pd
get_ipython().run_line_magic('pylab', 'inline')


# In[4]:


N_satellites_per_orbit = 10
N_orbits = 20
imaging_folder = '../Russia2Constellation/'
comm_folder = '../Facility2Constellation/'


# In[5]:


dataframes_imaging = []
for orbit in trange(N_orbits):
    with open(os.path.join(imaging_folder,"AreaTarget-Russia-To-Satellite-KinoSat_{:02d}_plane.txt".format(1+orbit)), 'r') as file:
        content = file.read()
    for satellite in range(N_satellites_per_orbit):
        # Extract the table
        table_start = content.index('Russia-To-KinoSat_11{:02d}{:02d}'.format(1+orbit,1+satellite))
        table_end = table_start + content[table_start:].index('\nMin')
        table_content = content[table_start:table_end].strip()

        # Create a list of lines in the table
        table_lines = table_content.split('\n')

        # Extract the table headers
        header_line = table_lines[2]
        headers = header_line.split()

        # Extract the table rows
        rows = [line.split() for line in table_lines[4:]]

        # Create a DataFrame from the table
        df = pd.DataFrame(rows)

        # Convert date columns to datetime format
        date_columns = [4,8]
        df[date_columns] = df[date_columns].apply(pd.to_datetime)
        df[4] = df[4].dt.time
        df[8] = df[8].dt.time
        dataframes_imaging.append(df)


# In[6]:


dataframes_imaging = []
for orbit in trange(N_orbits):
    with open(os.path.join(imaging_folder,"AreaTarget-Russia-To-Satellite-KinoSat_{:02d}_plane.txt".format(1+orbit)), 'r') as file:
        content = file.read()
    for satellite in range(N_satellites_per_orbit):
        # Extract the table
        table_start = content.index('Russia-To-KinoSat_11{:02d}{:02d}'.format(1+orbit,1+satellite))
        table_end = table_start + content[table_start:].index('\nMin')
        table_content = content[table_start:table_end].strip()

        # Create a list of lines in the table
        table_lines = table_content.split('\n')

        # Extract the table headers
        header_line = table_lines[2]
        headers = header_line.split()

        # Extract the table rows
        rows = [line.split() for line in table_lines[4:]]

        # Create a DataFrame from the table
        df = pd.DataFrame(rows)

        # Convert date columns to datetime format
        date_columns = [4,8]
        df[date_columns] = df[date_columns].apply(pd.to_datetime)
        df[4] = df[4].dt.time
        df[8] = df[8].dt.time
        dataframes_imaging.append(df)


# In[7]:


stations = [f.split('-')[1].split('.txt')[0] for f in os.listdir(comm_folder) if 'Facility' in f]
N_stations = len(stations)


# In[8]:


dataframes_comm = []
for i,station in tqdm(enumerate(stations)):
    with open(os.path.join(comm_folder,'Facility-{}.txt'.format(station)),'r') as file:
        content = file.read()
    for orbit in trange(N_orbits):
        for satellite in range(N_satellites_per_orbit):
            # Extract the table
            station_fix = station
            if station_fix == 'Cape_Town':
                station_fix = 'CapeTown'
            if station_fix == 'Dehli':
                station_fix = 'Delhi'
                
            table_start = content.index('{}-To-KinoSat_11{:02d}{:02d}'.format(station_fix,1+orbit,1+satellite))
            table_end = table_start + content[table_start:].index('\nMin')
            table_content = content[table_start:table_end].strip()

            # Create a list of lines in the table
            table_lines = table_content.split('\n')

            # Extract the table headers
            header_line = table_lines[2]
            headers = header_line.split()

            # Extract the table rows
            rows = [line.split() for line in table_lines[4:]]

            # Create a DataFrame from the table
            df = pd.DataFrame(rows)

            # Convert date columns to datetime format
            date_columns = [4,8]
            df[date_columns] = df[date_columns].apply(pd.to_datetime)
            df[4] = df[4].dt.time
            df[8] = df[8].dt.time
            dataframes_comm.append(df)


# In[9]:


dataframes_visible = dataframes_imaging.copy()


# In[10]:


df_filters = []
for df in tqdm(dataframes_visible):
    filtered_df = df[(df[4].astype(str).apply(pd.to_datetime).dt.hour >= 9) & (df[8].astype(str).apply(pd.to_datetime).dt.hour < 18)]
    df_filters.append(filtered_df)


# In[22]:


def create_imaging_array(df_imag,day):
    day = str(day)
    imaging = np.zeros([len(df_imag),86400])
    for i,df_input in tqdm(enumerate(df_imag)):
        result = np.zeros(86400, dtype=int)
        period_end = pd.to_timedelta(df_input[df_input[1]==day][8].astype(str).apply(pd.to_datetime).dt.time.astype(str)).dt.total_seconds().astype(int).values
        period_start = pd.to_timedelta(df_input[df_input[1]==day][4].astype(str).apply(pd.to_datetime).dt.time.astype(str)).dt.total_seconds().astype(int).values
        for r in range(len(period_end)):
            result[period_start[r]:period_end[r]] = 1
        imaging[i,:] = result
    return imaging
        
def create_downlink_array(df_imag,day):
    day = str(day)
    imaging = np.zeros([len(df_imag),86400])
    for i,df_input in tqdm(enumerate(df_imag)):
        result = np.zeros(86400, dtype=int)
        period_end = pd.to_timedelta(df_input[df_input[1]==day][8].astype(str).apply(pd.to_datetime).dt.time.astype(str)).dt.total_seconds().astype(int).values
        period_start = pd.to_timedelta(df_input[df_input[1]==day][4].astype(str).apply(pd.to_datetime).dt.time.astype(str)).dt.total_seconds().astype(int).values
        for r in range(len(period_end)):
            result[period_start[r]:period_end[r]] = 1
        imaging[i,:] = result
    imaging = imaging.reshape(N_stations,-1,86400).transpose(1,0,2)
    return imaging


# In[23]:


imaging = create_imaging_array(df_filters,1)
downlink = create_downlink_array(dataframes_comm,1)


# In[50]:


import numpy as np

def round_robin_scheduler_with_capacity(imaging, downlink, N, M):
    imaging_minutes = 86400

    # Step 1: Initialize variables
    memory = np.zeros(N, dtype=int)
    memory_per_second = 4
    throughput_per_second = 1
    imaging_schedule = np.zeros((N, imaging_minutes), dtype=int)
    downlink_schedule = np.zeros((N, M, imaging_minutes), dtype=int)
    satellite_last_downlink = np.zeros(N, dtype=int)
    ground_station_busy = np.zeros((M, imaging_minutes), dtype=bool)

    # Step 2: Iterate through each second
    for time in trange(imaging_minutes):
        # Step 3: For each satellite at the current time
        for satellite in range(N):
            # Step 3a: Check if satellite has available memory and can perform imaging
            if memory[satellite] < 8000 and imaging[satellite][time] == 1:
                imaging_schedule[satellite][time] = 1
                memory[satellite] += memory_per_second
            # Step 3b: If satellite's memory is full or not imaging, find next ground station for downlink
            elif memory[satellite] > 0:
                ground_station_found = False
                for i in range(M):
                    # Find the next ground station based on round-robin order
                    ground_station = (satellite_last_downlink[satellite] + i + 1) % M
                    if downlink[satellite][ground_station][time] == 1 and not ground_station_busy[ground_station][time]:
                        downlink_schedule[satellite][ground_station][time] = 1
                        memory[satellite] -= throughput_per_second
                        satellite_last_downlink[satellite] = ground_station
                        ground_station_busy[ground_station][time] = True
                        ground_station_found = True
                        break
                # Step 3c: If no suitable ground station found, move to the next satellite
                if not ground_station_found:
                    continue

    # Step 4: Return the imaging_schedule and downlink_schedule arrays
    return imaging_schedule, downlink_schedule, memory


import numpy as np
from ortools.linear_solver import pywraplp

def integer_linear_optimization(imaging, downlink, N, M):
    # Create linear programming solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    imaging_minutes = 86400
    x = {}
    y = {}

    # Define decision variables
    for satellite in range(N):
        for time in range(imaging_minutes):
            x[satellite, time] = solver.BoolVar(f'x[{satellite}, {time}]')
            for ground_station in range(M):
                y[satellite, ground_station, time] = solver.BoolVar(f'y[{satellite}, {ground_station}, {time}]')

    # Set objective function
    objective = solver.Objective()
    for satellite in range(N):
        for time in range(imaging_minutes):
            objective.SetCoefficient(x[satellite, time], 1)
    objective.SetMaximization()

    # Add constraints
    for satellite in range(N):
        for time in range(imaging_minutes):
            # Constraint for simultaneous imaging and transmission avoidance
            for ground_station in range(M):
                constraint1 = solver.Constraint(0, 1)
                constraint1.SetCoefficient(x[satellite, time], 1)
                constraint1.SetCoefficient(y[satellite, ground_station, time], 1)

            # Constraint for satellite memory
            constraint2 = solver.Constraint(-solver.infinity(), 0)
            constraint2.SetCoefficient(x[satellite, time], -1000)
            for ground_station in range(M):
                constraint2.SetCoefficient(y[satellite, ground_station, time], 8000)

            # Constraint for imaging feasibility
            if imaging[satellite][time] == 0:
                constraint3 = solver.Constraint(0, 0)
                constraint3.SetCoefficient(x[satellite, time], 1)

            # Constraint for downlink feasibility
            for ground_station in range(M):
                if downlink[satellite][ground_station][time] == 0:
                    constraint4 = solver.Constraint(0, 0)
                    constraint4.SetCoefficient(y[satellite, ground_station, time], 1)

    # Solve the linear program
    solver.parameters.max_time_in_seconds = 30.0
    solver.Solve()

    # Extract solution values
    imaging_schedule = np.zeros((N, imaging_minutes), dtype=int)
    downlink_schedule = np.zeros((N, M, imaging_minutes), dtype=int)
    for satellite in range(N):
        for time in range(imaging_minutes):
            imaging_schedule[satellite][time] = int(x[satellite, time].solution_value())
            for ground_station in range(M):
                downlink_schedule[satellite][ground_station][time] = int(y[satellite, ground_station, time].solution_value())

    # Return imaging_schedule and downlink_schedule arrays
    return imaging_schedule, downlink_schedule


# In[51]:


#INITIAL ROUND ROBIN SCHEDULER
imaging_schedule,downlink_schedule,memory = round_robin_scheduler_with_capacity(imaging,downlink,200,14)


# In[25]:


#ILP SOLVER
imaging_schedule,downlink_schedule = integer_linear_optimization(imaging,downlink,200,14)


# In[73]:


total_dict = []
for i,station_name in tqdm(enumerate(stations)):
    for j in range(40000,43000):
        if downlink_schedule[:,i,j].sum()>0:
            dict_station = {}
            sat_num = np.argmax(downlink_schedule[:,i,j])
            sat = '11{:02d}{:02d}'.format(sat_num//10+1,sat_num%10+1)
            dict_station['sat'] = sat
            dict_station['station'] = station_name
            dict_station['income_data'] = 125.0
            dict_station['start'] = str(datetime.timedelta(seconds=j))
            dict_station['end'] = str(datetime.timedelta(seconds=j+1))
            total_dict.append(dict_station)


# In[74]:


with open("satellite_small.json", "w") as outfile:
    json.dump(total_dict, outfile)


# In[69]:


import json 

imaging_schedule,downlink_schedule,memory = round_robin_scheduler_with_capacity(imaging,downlink,200,14)

total_dict = []
for i,station_name in tqdm(enumerate(stations)):
    for j in range(86400):
        if downlink_schedule[:,i,j].sum()>0:
            dict_station = {}
            sat_num = np.argmax(downlink_schedule[:,i,j])
            sat = '11{:02d}{:02d}'.format(sat_num//10+1,sat_num%10+1)
            dict_station['sat'] = sat
            dict_station['station'] = station_name
            dict_station['income_data'] = 125.0
            dict_station['start'] = str(datetime.timedelta(seconds=j))
            dict_station['end'] = str(datetime.timedelta(seconds=j+1))
            total_dict.append(dict_station)
            
with open("satellite.json", "w") as outfile:
    json.dump(total_dict, outfile)

