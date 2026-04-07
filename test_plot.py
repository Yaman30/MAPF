import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
import casadi as ca
import json

# 1. INSTÄLLNINGAR

L = 1.0
W = 0.5
Ts = 0.1

# Fysiska gränser
v_max = 2.0
a_max = 0.5
delta_max = 0.6

# Kollision
COLLISION_RADIUS = 1.0
SAFE_ZONE_RADIUS = 1.5

# MPC
N_horizon = 15




# 2. REFERENSBANA FRÅN JSON


def bygg_segment_fran_schema(schema_agent, noder):
    segments = []
    last_psi = 0.0

    for i, event in enumerate(schema_agent):
        handling = event[0]

        if isinstance(handling, str):
            node = handling
            t0 = float(event[1])
            t1 = float(event[2])
            segments.append({
                "type": "hold",
                "node": node,
                "t0": t0,
                "t1": t1,
                "psi": last_psi
            })

        elif isinstance(handling, list) and len(handling) == 2:
            node_a, node_b = handling
            t0 = float(event[1])
            t1 = float(schema_agent[i + 1][1]) if i + 1 < len(schema_agent) else t0 + 1.0

            p0 = np.array(noder[node_a], dtype=float)
            p1 = np.array(noder[node_b], dtype=float)

            d = p1 - p0
            if np.linalg.norm(d) > 1e-9:
                psi = float(np.arctan2(d[1], d[0]))
                last_psi = psi
            else:
                psi = last_psi

            segments.append({
                "type": "move",
                "from": node_a,
                "to": node_b,
                "t0": t0,
                "t1": t1,
                "psi": psi
            })

    return segments


def get_reference_at_time(robot, t):
    last_seg = None
    for seg in robot.segments:
        last_seg = seg
        if seg["t0"] <= t <= seg["t1"]:
            if seg["type"] == "hold":
                x, y = robot.noder[seg["node"]]
                return float(x), float(y), float(seg["psi"]), 0.0

            elif seg["type"] == "move":
                p0 = np.array(robot.noder[seg["from"]], dtype=float)
                p1 = np.array(robot.noder[seg["to"]], dtype=float)

                duration = max(seg["t1"] - seg["t0"], Ts)
                alpha = np.clip((t - seg["t0"]) / duration, 0.0, 1.0)
                p = (1.0 - alpha) * p0 + alpha * p1

                dist = np.linalg.norm(p1 - p0)
                v = min(dist / duration, v_max)

                return float(p[0]), float(p[1]), float(seg["psi"]), float(v)

    if last_seg is None:
        return 0.0, 0.0, 0.0, 0.0

    if last_seg["type"] == "hold":
        x, y = robot.noder[last_seg["node"]]
    else:
        x, y = robot.noder[last_seg["to"]]

    return float(x), float(y), float(last_seg["psi"]), 0.0


def build_reference_matrix(robot, current_time):
    ref_matrix = np.zeros((4, N_horizon + 1))
    for k in range(N_horizon + 1):
        t_k = current_time + k * Ts
        x_ref, y_ref, psi_ref, v_ref = get_reference_at_time(robot, t_k)
        ref_matrix[:, k] = [x_ref, y_ref, psi_ref, v_ref]
    return ref_matrix


# 3. ROBOTKLASS


class Robot:
    def __init__(self, id_num, start_state, agent_id, schema, noder, color, label):
        self.id = id_num
        self.agent_id = agent_id
        self.state = np.array([*start_state, 0.0], dtype=float)
        self.color = color
        self.label = label
        self.noder = noder

        self.schema = schema[agent_id]
        self.segments = bygg_segment_fran_schema(self.schema, self.noder)

        self.x_hist = [self.state[0]]
        self.y_hist = [self.state[1]]
        self.state_hist = []
        self.control_hist = []
        self.finished = False

        # Start- och slutmål
        self.start_xy = np.array(start_state[:2], dtype=float)
        goal_x, goal_y, _, _ = self.get_final_goal()
        self.goal_xy = np.array([goal_x, goal_y], dtype=float)

    def get_final_goal(self):
        last_seg = self.segments[-1]
        if last_seg["type"] == "hold":
            x, y = self.noder[last_seg["node"]]
        else:
            x, y = self.noder[last_seg["to"]]
        return x, y, last_seg["psi"], 0.0



# 4. LÄS IN JSON


with open(r'C:\PyCharm\MAPF\Instances\maps\sets\set_0\instance_25254ef2-0876-4627-80d9-0c97b76cfbe9.json', 'r') as f:
    karta = json.load(f)

with open(r'C:\PyCharm\MAPF\Instances\solutions\sets\set_0\instance_25254ef2-0876-4627-80d9-0c97b76cfbe.json', 'r') as f:
    schema = json.load(f)

noder = {nod['id']: nod['pos'] for nod in karta['graph']['nodes']}
robot_namn = list(schema.keys())

robots = []
färger = ['cyan', 'orange', 'green', 'purple', 'red', 'blue', 'pink', 'brown', 'gray', 'olive']

for i, agent_id in enumerate(robot_namn):
    start_nod = karta['agent_start'][agent_id]
    start_x, start_y = noder[start_nod]
    start_state = [start_x, start_y, 0.0]
    färg = färger[i % len(färger)]
    robots.append(Robot(i + 1, start_state, agent_id, schema, noder, färg, f"Robot {agent_id}"))



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

        cost += 300.0 * (X[0, k] - X_ref[0, k]) ** 2
        cost += 300.0 * (X[1, k] - X_ref[1, k]) ** 2
        cost += 20.0  * (X[2, k] - X_ref[2, k]) ** 2
        cost += 10.0  * (X[3, k] - X_ref[3, k]) ** 2

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


fig, ax = plt.subplots(figsize=(20,20))
ax.set_xlim(-2, 115)
ax.set_ylim(-2, 30)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Strikt JSON-spårning med central säkerhetszon")

gfx_objects = {}

for r in robots:
    rect = patches.Rectangle((0, 0), L, W, color=r.color, alpha=0.8, label=r.label)
    trail, = ax.plot([], [], color=r.color, linewidth=2)

    safe_zone = patches.Circle(
        (r.state[0], r.state[1]),
        SAFE_ZONE_RADIUS,
        fill=False,
        linestyle='--',
        linewidth=1.5,
        edgecolor=r.color,
        alpha=0.35
    )

    start_marker, = ax.plot(
        [r.start_xy[0]], [r.start_xy[1]],
        marker='o', linestyle='None', color=r.color, markersize=7, alpha=0.9
    )
    goal_marker, = ax.plot(
        [r.goal_xy[0]], [r.goal_xy[1]],
        marker='x', linestyle='None', color=r.color, markersize=9, alpha=0.9
    )

    ax.add_patch(rect)
    ax.add_patch(safe_zone)

    gfx_objects[r.id] = {
        'rect': rect,
        'trail': trail,
        'safe_zone': safe_zone,
        'start_marker': start_marker,
        'goal_marker': goal_marker
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

    # Kollisioner räknas, men robotarna stoppas inte
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
        if not r.finished:
            ref_matrix = build_reference_matrix(r, current_time)

            opti, X0_p, X_ref_p, U_v = mpc_solvers[r.id]
            opti.set_value(X0_p, r.state)
            opti.set_value(X_ref_p, ref_matrix)

            try:
                sol = opti.solve()
                u_opt = sol.value(U_v[:, 0])
            except RuntimeError:
                u_opt = opti.debug.value(U_v[:, 0])

            a_cmd, delta_cmd = u_opt[0], u_opt[1]
            r.control_hist.append([a_cmd, delta_cmd])

            st = r.state
            k1 = robot_dynamics(st, a_cmd, delta_cmd)
            k2 = robot_dynamics(st + Ts / 2 * k1, a_cmd, delta_cmd)
            k3 = robot_dynamics(st + Ts / 2 * k2, a_cmd, delta_cmd)
            k4 = robot_dynamics(st + Ts * k3, a_cmd, delta_cmd)

            r.state = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            r.state[3] = np.clip(r.state[3], 0.0, v_max)

            r.state_hist.append(r.state.copy())
            r.x_hist.append(r.state[0])
            r.y_hist.append(r.state[1])

            # Nått slutmålet?
            goal_x, goal_y, _, _ = r.get_final_goal()
            goal_dist = np.hypot(r.state[0] - goal_x, r.state[1] - goal_y)

            if goal_dist < 0.35:
                r.finished = True
                r.state[3] = 0.0

        x, y, psi, v = r.state
        gfx = gfx_objects[r.id]

        diag = np.sqrt(L ** 2 + W ** 2) / 2
        q = np.arctan(W / L) + psi
        gfx['rect'].set_xy([x - diag * np.cos(q), y - diag * np.sin(q)])
        gfx['rect'].angle = np.degrees(psi)
        gfx['trail'].set_data(r.x_hist, r.y_hist)

        # Säkerhetszonen
        gfx['safe_zone'].center = (x, y)

        artists.extend([
            gfx['rect'],
            gfx['trail'],
            gfx['safe_zone'],
            gfx['start_marker'],
            gfx['goal_marker']
        ])

    return artists


ani = FuncAnimation(fig, update, frames=1000, interval=Ts * 1000, blit=False)
plt.legend(loc='upper right')
#ani.save('mpc_robot_ghost_reference.mp4', writer='ffmpeg', fps=30, dpi=100)
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

ax_states[-1].set_xlabel("time")
ax_ctrls[-1].set_xlabel("time")

ax_states[0].legend()
ax_ctrls[0].legend()

plt.show()