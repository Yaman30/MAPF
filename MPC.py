import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
import casadi as ca

from ReferenceTrajectoryUtils import get_instance_trajectories

# 1. INSTÄLLNINGAR
L = 1.0
W = 0.5
Ts = 0.1
v_max = 2.0
a_max = 1.5
delta_max = 1.0
COLLISION_RADIUS = 1.0
SAFE_ZONE_RADIUS = 1.5
N_horizon = 15
LOOKAHEAD_TIME = 0.3
GHOST_VISUAL_TIME = 1.0

# Instansnamn
INSTANCE_NAME = 'instance_2f9c3898-f1c5-4cb2-81ec-9f9745c0f2a2.json'

# 2. REFERENSBANA

def unwrap_angle_near_reference(angle, reference):
    while angle - reference > np.pi: angle -= 2 * np.pi
    while angle - reference < -np.pi: angle += 2 * np.pi
    return angle

def build_time_based_reference_matrix(robot, current_time):
    ref_matrix = np.zeros((4, N_horizon + 1))
    current_psi = robot.state[2]

    for k in range(N_horizon + 1):
        t_k = current_time + LOOKAHEAD_TIME + (k * Ts)

        ref_state = robot.trajectory(t_k)
        x_ref, y_ref, psi_ref_raw, v_ref = ref_state[0], ref_state[1], ref_state[2], ref_state[3]

        if v_ref < 0.05:
            psi_ref = current_psi
        else:
            psi_ref = unwrap_angle_near_reference(psi_ref_raw, current_psi)

        ref_matrix[:, k] = [x_ref, y_ref, psi_ref, v_ref]
        current_psi = psi_ref

    # SKAPA GRAFISK
    ghost_t = current_time + GHOST_VISUAL_TIME
    ghost_state = robot.trajectory(ghost_t)

    return ref_matrix, [ghost_state[0], ghost_state[1]]

# 3. ROBOTKLASS
class Robot:
    def __init__(self, id_num, agent_id, trajectory_fn, color, label):
        self.id = id_num
        self.agent_id = agent_id

        self.trajectory = trajectory_fn

        start_ref = self.trajectory(0.0)
        self.state = np.array([start_ref[0], start_ref[1], start_ref[2], 0.0], dtype=float)

        self.color = color
        self.label = label

        self.x_hist = [self.state[0]]
        self.y_hist = [self.state[1]]
        self.state_hist = []
        self.control_hist = []
        self.finished = False

        self.start_xy = np.array([start_ref[0], start_ref[1]], dtype=float)

        final_ref = self.trajectory(10000.0)
        self.goal_xy = np.array([final_ref[0], final_ref[1]], dtype=float)
        self.goal_psi = final_ref[2]

    def get_final_goal(self):
        return self.goal_xy[0], self.goal_xy[1], self.goal_psi, 0.0

# 4. LÄS IN INSTANS OCH SKAPA ROBOTAR
trajectories = get_instance_trajectories(INSTANCE_NAME, wait_dir='incoming')

robots = []
färger = ['cyan', 'orange', 'green', 'purple', 'red', 'blue', 'pink', 'brown', 'gray', 'olive']

for i, (agent_id, traj_fn) in enumerate(trajectories.items()):
    färg = färger[i % len(färger)]
    robots.append(Robot(i + 1, agent_id, traj_fn, färg, f"Robot {agent_id}"))

# 5. MPC

def create_mpc_solver():
    opti = ca.Opti()
    X = opti.variable(4, N_horizon + 1)
    U = opti.variable(2, N_horizon)
    X0 = opti.parameter(4)
    X_ref = opti.parameter(4, N_horizon + 1)

    opti.subject_to(X[:, 0] == X0)

    state_sym = ca.SX.sym('x', 4)
    ctrl_sym = ca.SX.sym('u', 2)
    v_s, psi_s = state_sym[3], state_sym[2]
    a_s, delta_s = ctrl_sym[0], ctrl_sym[1]

    dx = v_s * ca.cos(psi_s)
    dy = v_s * ca.sin(psi_s)
    dpsi = (v_s / L) * ca.tan(delta_s)
    dv = a_s

    f_kin = ca.Function('f_kin', [state_sym, ctrl_sym], [ca.vertcat(dx, dy, dpsi, dv)])
    cost = 0

    for k in range(N_horizon):
        st, con = X[:, k], U[:, k]
        k1 = f_kin(st, con)
        k2 = f_kin(st + Ts / 2 * k1, con)
        k3 = f_kin(st + Ts / 2 * k2, con)
        k4 = f_kin(st + Ts * k3, con)
        x_next_rk4 = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        opti.subject_to(X[:, k + 1] == x_next_rk4)

        cost += 150.0 * (X[0, k] - X_ref[0, k]) ** 2
        cost += 150.0 * (X[1, k] - X_ref[1, k]) ** 2
        cost += 80.0 * (X[2, k] - X_ref[2, k]) ** 2
        cost += 10.0 * (X[3, k] - X_ref[3, k]) ** 2
        cost += 1.0 * U[0, k] ** 2
        cost += 5.0 * U[1, k] ** 2

    opti.subject_to(opti.bounded(-a_max, U[0, :], a_max))
    opti.subject_to(opti.bounded(-delta_max, U[1, :], delta_max))
    opti.subject_to(opti.bounded(0, X[3, :], v_max))

    opti.minimize(cost)
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})

    return opti, X0, X_ref, U


mpc_solvers = {r.id: create_mpc_solver() for r in robots}


def robot_dynamics(st, a, delta):
    dx = st[3] * np.cos(st[2])
    dy = st[3] * np.sin(st[2])
    dpsi = (st[3] / L) * np.tan(delta)
    dv = a
    return np.array([dx, dy, dpsi, dv])

# 6. GRAFIK OCH ANIMERING

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_xlim(-2, 85)
ax.set_ylim(-2, 50)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("MPC + JSON")

gfx_objects = {}

for r in robots:
    rect = patches.Rectangle((0, 0), L, W, color=r.color, alpha=0.8, label=r.label)
    trail, = ax.plot([], [], color=r.color, linewidth=2)

    safe_zone = patches.Circle(
        (r.state[0], r.state[1]), SAFE_ZONE_RADIUS, fill=False,
        linestyle='--', linewidth=1.5, edgecolor=r.color, alpha=0.35
    )

    start_marker, = ax.plot([r.start_xy[0]], [r.start_xy[1]], marker='o', color=r.color, markersize=7)
    goal_marker, = ax.plot([r.goal_xy[0]], [r.goal_xy[1]], marker='x', color=r.color, markersize=9)
    ghost_marker, = ax.plot([], [], marker='*', color=r.color, markersize=10, linestyle='None')

    ax.add_patch(rect)
    ax.add_patch(safe_zone)

    gfx_objects[r.id] = {
        'rect': rect, 'trail': trail, 'safe_zone': safe_zone,
        'start_marker': start_marker, 'goal_marker': goal_marker, 'ghost_marker': ghost_marker
    }

status_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, color='red', fontweight='bold')
time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=12, fontweight='bold')

collision_count = 0
active_collisions = set()


def update(frame):
    global collision_count

    current_time = frame * Ts
    artists = [status_text, time_text]
    time_text.set_text(f"Global Tid: {current_time:.1f}s")

    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            r1 = robots[i]
            r2 = robots[j]
            dist = np.hypot(r1.state[0] - r2.state[0], r1.state[1] - r2.state[1])
            pair = (min(r1.id, r2.id), max(r1.id, r2.id))

            if dist < 2 * COLLISION_RADIUS:
                if pair not in active_collisions:
                    collision_count += 1
                    active_collisions.add(pair)
            else:
                if pair in active_collisions:
                    active_collisions.remove(pair)

    status_text.set_text(f"Antal kollisioner: {collision_count}")

    for r in robots:
        gfx = gfx_objects[r.id]

        if not r.finished:
            ref_matrix, ghost_state = build_time_based_reference_matrix(r, current_time)
            gfx['ghost_marker'].set_data([ghost_state[0]], [ghost_state[1]])

            opti, X0_p, X_ref_p, U_v = mpc_solvers[r.id]
            opti.set_value(X0_p, r.state)
            opti.set_value(X_ref_p, ref_matrix)

            try:
                sol = opti.solve()
                u_opt = sol.value(U_v[:, 0])
            except RuntimeError:
                u_opt = np.array([-a_max, 0.0])

            a_cmd, delta_cmd = u_opt[0], u_opt[1]
            r.control_hist.append([a_cmd, delta_cmd])

            st = r.state.copy()
            k1 = robot_dynamics(st, a_cmd, delta_cmd)
            k2 = robot_dynamics(st + Ts / 2 * k1, a_cmd, delta_cmd)
            k3 = robot_dynamics(st + Ts / 2 * k2, a_cmd, delta_cmd)
            k4 = robot_dynamics(st + Ts * k3, a_cmd, delta_cmd)

            r.state = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            r.state[3] = np.clip(r.state[3], 0.0, v_max)

            while r.state[2] > np.pi: r.state[2] -= 2 * np.pi
            while r.state[2] < -np.pi: r.state[2] += 2 * np.pi

            r.state_hist.append(r.state.copy())
            r.x_hist.append(r.state[0])
            r.y_hist.append(r.state[1])

            goal_x, goal_y, _, _ = r.get_final_goal()
            goal_dist = np.hypot(r.state[0] - goal_x, r.state[1] - goal_y)

            if goal_dist < 0.6:
                r.finished = True
                r.state[3] = 0.0
                gfx['ghost_marker'].set_data([], [])
        else:
            gfx['ghost_marker'].set_data([], [])

        x, y, psi, v = r.state

        diag = np.sqrt(L ** 2 + W ** 2) / 2
        q = np.arctan(W / L) + psi
        gfx['rect'].set_xy([x - diag * np.cos(q), y - diag * np.sin(q)])
        gfx['rect'].angle = np.degrees(psi)
        gfx['trail'].set_data(r.x_hist, r.y_hist)
        gfx['safe_zone'].center = (x, y)

        artists.extend([
            gfx['rect'], gfx['trail'], gfx['safe_zone'],
            gfx['start_marker'], gfx['goal_marker'], gfx['ghost_marker']
        ])

    return artists


ani = FuncAnimation(fig, update, frames=1000, interval=Ts * 1000, blit=False)
plt.legend(loc='upper right')
ani.save('mpc_robot_ghost_reference.mp4', writer='ffmpeg', fps=30, dpi=100)
plt.show()

# 7. STATES / CONTROLS / STATISTIK

fig_states, ax_states = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
fig_ctrls, ax_ctrls = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

labels_states = ['x', 'y', 'psi', 'v']
labels_ctrls = ['a', 'delta']

for r in robots:
    s = np.array(r.state_hist)
    u = np.array(r.control_hist)

    if len(s) > 0:
        t_s = np.arange(len(s)) * Ts
        for i in range(4):
            ax_states[i].plot(t_s, s[:, i], label=r.label)
            ax_states[i].set_ylabel(labels_states[i])
            ax_states[i].grid(True)

    if len(u) > 0:
        t_u = np.arange(len(u)) * Ts
        for i in range(2):
            ax_ctrls[i].plot(t_u, u[:, i], label=r.label)
            ax_ctrls[i].set_ylabel(labels_ctrls[i])
            ax_ctrls[i].grid(True)

ax_states[-1].set_xlabel("time (s)")
ax_ctrls[-1].set_xlabel("time (s)")

handles, labels = ax_states[0].get_legend_handles_labels()
if labels:
    ax_states[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax_ctrls[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1))

plt.show()