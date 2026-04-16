import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

MAPS_DIR = 'Instances/maps'
SOLUTIONS_DIR = 'Instances/solutions'

def get_instance_paths(instance):

    # Search through maps and solutions folders to find the map and solution file paths
    map_sets_folder = os.path.join(MAPS_DIR, 'sets')
    sol_sets_folder = os.path.join(SOLUTIONS_DIR, 'sets')
    map_path = None
    sol_path = None

    # - maps
    for dir in os.listdir(map_sets_folder):
        folder_path = os.path.join(map_sets_folder, dir)
        if not os.path.isdir(folder_path):
            continue

        # iterate over files in the directory
        for file in os.listdir(folder_path):
            if file == instance:
                map_path = os.path.join(folder_path, file)
                break
    for dir in os.listdir(sol_sets_folder):
        folder_path = os.path.join(sol_sets_folder, dir)
        if not os.path.isdir(folder_path):
            continue

        # iterate over files in the directory
        for file in os.listdir(folder_path):
            if file == instance:
                sol_path = os.path.join(folder_path, file)
                break

    if map_path is None:
        raise FileNotFoundError(f'Could not find map file for instance {instance}')
    if sol_path is None:
        raise FileNotFoundError(f'Could not find solution file for instance {instance}')

    return map_path, sol_path

def read_map_json(map_path):

    # read the map json file and return the map data
    with open(map_path, 'r') as f:
        map_data = json.load(f)

    graph = json_graph.node_link_graph(map_data['graph'])
    return graph

def read_sol_json(sol_path):

    # read the solution json file and return the solution data
    with open(sol_path, 'r') as f:
        sol_data = json.load(f)

    return sol_data

def get_instance_trajectories(instance, wait_dir='incoming'):

    assert wait_dir in ['incoming', 'average', 'outgoing'], f'The wait direction must be one of (incoming, average, outgoing).'

    map_path, sol_path = get_instance_paths(instance)

    graph = read_map_json(map_path)
    paths = read_sol_json(sol_path)

    trajectory_functions = dict()
    for agent, path in paths.items():

        directions = [None for _ in range(len(path))]
        speeds = [None for _ in range(len(path))]

        # 1st pass: go through each action, and if move action, compute direction and speed
        for i, action in enumerate(path):

            # if wait action, skip
            if len(action) == 3:
                continue

            start_node = action[0][0]
            end_node = action[0][1]

            start_pos = np.array(graph.nodes[start_node]['pos'])
            end_pos = np.array(graph.nodes[end_node]['pos'])
            edge_time_to_traverse = graph.edges[start_node, end_node]['weight']

            direction = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            speed = np.linalg.norm(end_pos - start_pos) / edge_time_to_traverse

            directions[i] = direction
            speeds[i] = speed

        # 2nd pass: go through each action, and if wait action, fill in direction (between previous and next move action)
        for i, action in enumerate(path):

            # if move action, skip
            if len(action) == 2:
                continue

            # if first action, direction is the same as the next action
            if i == 0 and i == len(path) - 1:
                direction = 0.0  # single wait action with no neighbors
            elif i == 0:
                direction = directions[i+1]
            # if last action, direction is the same as the previous action
            elif i == len(path) - 1:
                direction = directions[i-1]
            else:
                dir_prev = directions[i - 1]
                dir_next = directions[i + 1]

                if wait_dir == 'incoming':
                    # direction is the same as the previous action
                    direction = dir_prev

                elif wait_dir == 'average':
                    # direction is the average of the previous and next action
                    avg_sin = (np.sin(dir_prev) + np.sin(dir_next)) / 2.0
                    avg_cos = (np.cos(dir_prev) + np.cos(dir_next)) / 2.0
                    direction = np.arctan2(avg_sin, avg_cos)

                elif wait_dir == 'outgoing':
                    # direction is the same as the next action
                    direction = dir_next

            directions[i] = direction
            speeds[i] = 0

        assert None not in directions, f'Error: agent {agent} has None direction in its trajectory'
        assert None not in speeds, f'Error: agent {agent} has None speed in its trajectory'

        # construct trajectory function components
        component_functions = [None for _ in range(len(path))]
        for i, action in enumerate(path):

            # if wait action
            if len(action) == 3:

                static_state = np.array(graph.nodes[action[0]]['pos'] + [directions[i]] + [speeds[i]])
                component_functions[i] = lambda t, s=static_state: s

            # if move action
            elif len(action) == 2:

                start_node = action[0][0]
                end_node = action[0][1]
                traversal_time = graph.edges[start_node, end_node]['weight']
                start_time = action[1]

                start_pos = np.array(graph.nodes[start_node]['pos'])
                end_pos = np.array(graph.nodes[end_node]['pos'])

                component_functions[i] = lambda t, s_pos=start_pos, e_pos=end_pos, s_time=start_time, tt=traversal_time, d=directions[i], s=speeds[i]: (
                    np.concatenate((s_pos + (t - s_time) / tt * (e_pos - s_pos), [d], [s]))
                )

        assert None not in component_functions, f'Error: agent {agent} has None component function in its trajectory'

        # put together components into the single, complete trajectory function for the agent
        action_time_intervals = [None for _ in range(len(path))]
        for i, action in enumerate(path):

            if len(action) == 3:  # wait action
                start_time = action[1]
                end_time = action[2]
            elif len(action) == 2:  # move action
                start_time = action[1]
                end_time = action[1] + graph.edges[action[0][0], action[0][1]]['weight']

            action_time_intervals[i] = (start_time, end_time)

        def trajectory(t, cfs=component_functions, p=path, timing=action_time_intervals):

            assert t >= 0, f'Error: time t must be non-negative, but got {t}'

            i = next((j for j, (start, end) in enumerate(timing) if start <= t < end), len(p)-1)

            state = cfs[i](t)

            # special case: if t >= end time of last action, then return the final state
            if i == len(p) - 1 and t >= timing[-1][1]:
                state = cfs[-1](timing[-1][1])

            return state

        trajectory_functions[agent] = trajectory

    return trajectory_functions


if __name__ == '__main__':

    # example of how to use:
    instance = 'instance_25254ef2-0876-4627-80d9-0c97b76cfbe9.json'
    trajectories = get_instance_trajectories(instance, wait_dir='incoming')

    # Trajectories (dict) contains one trajectory for each agent
    agent_0_trajectory = trajectories['a0']
    print(agent_0_trajectory(0))        # trajectory point at time t=0
    print(agent_0_trajectory(56.7))     # trajectory point at time t=56.7
    print(agent_0_trajectory(100))      # trajectory point at time t=100

