import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import casadi as ca

# 1. PARAMETRAR
Ts = 0.2
N = 10
L = 0.6
W = 0.3


obs_radius = 0.5
safe_radius = 0.6
alpha_cbf = 0.5

noder = {
    "v0": np.array([0.0, 0.0]),
    "v15": np.array([10.0, 10.0])
}

schema = [
    ["v0", 0.0, 2.0],
    [["v0", "v15"], 2.0, 15.0],
    ["v15", 15.0, 20.0]
]

schema_r2 = [
    ["v15", 0.0, 2.0],
    [["v15", "v0"], 2.0, 15.0],
    ["v0", 15.0, 20.0]
]


# 2. REFERENSGENERATORER
def get_ref_from_schema(t, schema_in):
    for i, h in enumerate(schema_in):
        handling = h[0]
        t_start, t_slut = h[1], h[2]

        if t_start <= t <= t_slut:
            if isinstance(handling, str):
                pos = noder[handling]

                # FIX 1: Räkna ut rätt vinkel (psi) redan när de står stilla
                psi = 0.0
                if i + 1 < len(schema_in) and isinstance(schema_in[i + 1][0], list):
                    n0 = noder[schema_in[i + 1][0][0]]
                    n1 = noder[schema_in[i + 1][0][1]]
                    psi = np.arctan2(n1[1] - n0[1], n1[0] - n0[0])
                elif i > 0 and isinstance(schema_in[i - 1][0], list):
                    n0 = noder[schema_in[i - 1][0][0]]
                    n1 = noder[schema_in[i - 1][0][1]]
                    psi = np.arctan2(n1[1] - n0[1], n1[0] - n0[0])

                return np.array([pos[0], pos[1], psi, 0.0])

            elif isinstance(handling, list):
                n0 = noder[handling[0]]
                n1 = noder[handling[1]]

                dt = t_slut - t_start
                dist = np.linalg.norm(n1 - n0)
                v_req = dist / dt

                procent = (t - t_start) / dt
                x = n0[0] + procent * (n1[0] - n0[0])
                y = n0[1] + procent * (n1[1] - n0[1])
                psi = np.arctan2(n1[1] - n0[1], n1[0] - n0[0])

                return np.array([x, y, psi, v_req])

    sista = schema_in[-1][0]
    sista_nod = noder[sista]


    psi_slut = 0.0
    if isinstance(schema_in[-2][0], list):
        n0 = noder[schema_in[-2][0][0]]
        n1 = noder[schema_in[-2][0][1]]
        psi_slut = np.arctan2(n1[1] - n0[1], n1[0] - n0[0])

    return np.array([sista_nod[0], sista_nod[1], psi_slut, 0.0])


def get_ref_at_time(t):
    return get_ref_from_schema(t, schema)


def get_ref_at_time_r2(t):
    return get_ref_from_schema(t, schema_r2)

def build_reference_path(ref_func, t_start=0.0, t_end=20.0, dt=0.02):
    t_grid = np.arange(t_start, t_end + dt, dt)
    path = np.array([ref_func(t) for t in t_grid])

    xy = path[:, :2]
    ds = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    s = np.concatenate(([0.0], np.cumsum(ds)))

    return path, s


ref_path_1, ref_s_1 = build_reference_path(get_ref_at_time)
ref_path_2, ref_s_2 = build_reference_path(get_ref_at_time_r2)


# Gemensamma fysik-funktioner
def f_kin(x, u):
    return ca.vertcat(
        x[3] * ca.cos(x[2]),
        x[3] * ca.sin(x[2]),
        (x[3] / L) * ca.tan(u[1]),
        u[0]
    )


R_safe = (safe_radius + obs_radius) ** 2


def h_cbf(x, obs_pos):
    return (x[0] - obs_pos[0]) ** 2 + (x[1] - obs_pos[1]) ** 2 - R_safe


def h_dot_cbf(x, obs_pos, obs_vel):
    x_dot = x[3] * ca.cos(x[2])
    y_dot = x[3] * ca.sin(x[2])

    x_o_dot = obs_vel[0]
    y_o_dot = obs_vel[1]

    return 2 * (x[0] - obs_pos[0]) * (x_dot - x_o_dot) + \
        2 * (x[1] - obs_pos[1]) * (y_dot - y_o_dot)


# 3. MPC FÖR ROBOT 1
opti_mpc = ca.Opti()

X = opti_mpc.variable(4, N + 1)
U = opti_mpc.variable(2, N)
X0 = opti_mpc.parameter(4)
Xref = opti_mpc.parameter(4, N + 1)

obs_pos_mpc = opti_mpc.parameter(2)
obs_vel_mpc = opti_mpc.parameter(2)

opti_mpc.subject_to(X[:, 0] == X0)
cost_mpc = 0

for k in range(N):
    k1 = f_kin(X[:, k], U[:, k])
    k2 = f_kin(X[:, k] + Ts / 2 * k1, U[:, k])
    k3 = f_kin(X[:, k] + Ts / 2 * k2, U[:, k])
    k4 = f_kin(X[:, k] + Ts * k3, U[:, k])
    x_next = X[:, k] + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    opti_mpc.subject_to(X[:, k + 1] == x_next)

    cost_mpc += 10.0 * (X[0, k] - Xref[0, k]) ** 2
    cost_mpc += 10.0 * (X[1, k] - Xref[1, k]) ** 2
    cost_mpc += 20.0 * (X[3, k] - Xref[3, k]) ** 2
    cost_mpc += 0.1 * U[0, k] ** 2
    cost_mpc += 1.0 * U[1, k] ** 2

    obs_k = obs_pos_mpc + k * Ts * obs_vel_mpc
    obs_k1 = obs_pos_mpc + (k + 1) * Ts * obs_vel_mpc

    h_k = h_cbf(X[:, k], obs_k)
    h_k1 = h_cbf(X[:, k + 1], obs_k1)
    opti_mpc.subject_to(h_k1 - h_k >= -alpha_cbf * h_k)

opti_mpc.subject_to(opti_mpc.bounded(-1.0, U[0, :], 1.0))
opti_mpc.subject_to(opti_mpc.bounded(-0.6, U[1, :], 0.6))
opti_mpc.subject_to(opti_mpc.bounded(0.0, X[3, :], 2.0))

opti_mpc.minimize(cost_mpc)
opti_mpc.solver('ipopt', {'print_time': 0, 'ipopt.print_level': 0, 'ipopt.sb': 'yes'})

# 3B. MPC FÖR ROBOT 2
opti_mpc_r2 = ca.Opti()

X_r2 = opti_mpc_r2.variable(4, N + 1)
U_r2 = opti_mpc_r2.variable(2, N)
X0_r2 = opti_mpc_r2.parameter(4)
Xref_r2 = opti_mpc_r2.parameter(4, N + 1)

opti_mpc_r2.subject_to(X_r2[:, 0] == X0_r2)
cost_mpc_r2 = 0

for k in range(N):
    k1 = f_kin(X_r2[:, k], U_r2[:, k])
    k2 = f_kin(X_r2[:, k] + Ts / 2 * k1, U_r2[:, k])
    k3 = f_kin(X_r2[:, k] + Ts / 2 * k2, U_r2[:, k])
    k4 = f_kin(X_r2[:, k] + Ts * k3, U_r2[:, k])
    x_next = X_r2[:, k] + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    opti_mpc_r2.subject_to(X_r2[:, k + 1] == x_next)

    cost_mpc_r2 += 10.0 * (X_r2[0, k] - Xref_r2[0, k]) ** 2
    cost_mpc_r2 += 10.0 * (X_r2[1, k] - Xref_r2[1, k]) ** 2
    cost_mpc_r2 += 20.0 * (X_r2[3, k] - Xref_r2[3, k]) ** 2
    cost_mpc_r2 += 0.1 * U_r2[0, k] ** 2
    cost_mpc_r2 += 1.0 * U_r2[1, k] ** 2

opti_mpc_r2.subject_to(opti_mpc_r2.bounded(-1.0, U_r2[0, :], 1.0))
opti_mpc_r2.subject_to(opti_mpc_r2.bounded(-0.6, U_r2[1, :], 0.6))
opti_mpc_r2.subject_to(opti_mpc_r2.bounded(0.0, X_r2[3, :], 2.0))

opti_mpc_r2.minimize(cost_mpc_r2)
opti_mpc_r2.solver('ipopt', {'print_time': 0, 'ipopt.print_level': 0, 'ipopt.sb': 'yes'})

# 4. CBF / QP FÖR ROBOT 1
opti_qp = ca.Opti()

U_star = opti_qp.variable(2)
U_mpc_in = opti_qp.parameter(2)
X_curr = opti_qp.parameter(4)

obs_pos_qp = opti_qp.parameter(2)
obs_vel_qp = opti_qp.parameter(2)

cost_qp = (U_star[0] - U_mpc_in[0]) ** 2 + (U_star[1] - U_mpc_in[1]) ** 2

k1_c = f_kin(X_curr, U_star)
k2_c = f_kin(X_curr + Ts / 2 * k1_c, U_star)
k3_c = f_kin(X_curr + Ts / 2 * k2_c, U_star)
k4_c = f_kin(X_curr + Ts * k3_c, U_star)
X_next_c = X_curr + Ts / 6 * (k1_c + 2 * k2_c + 2 * k3_c + k4_c)

obs_next_c = obs_pos_qp + Ts * obs_vel_qp

h_curr = h_cbf(X_curr, obs_pos_qp)
h_next = h_cbf(X_next_c, obs_next_c)
h_dot_expr = h_dot_cbf(X_curr, obs_pos_qp, obs_vel_qp)

opti_qp.subject_to(h_next - h_curr >= -alpha_cbf * h_curr)

opti_qp.subject_to(opti_qp.bounded(-1.0, U_star[0], 1.0))
opti_qp.subject_to(opti_qp.bounded(-0.6, U_star[1], 0.6))
opti_qp.subject_to(X_curr[3] + Ts * U_star[0] >= 0.0)

opti_qp.minimize(cost_qp)
opti_qp.solver('ipopt', {'print_time': 0, 'ipopt.print_level': 0, 'ipopt.sb': 'yes'})

# 5. SIMULERING & GRAFIK
state = get_ref_at_time(0.0)
traj_x, traj_y = [state[0]], [state[1]]
state_hist = [state.copy()]
control_hist = []

state_r2 = get_ref_at_time_r2(0.0)
r2_state_hist = [state_r2.copy()]
r2_control_hist = [np.array([0.0, 0.0])]
r2_traj_x, r2_traj_y = [state_r2[0]], [state_r2[1]]

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 12)
ax.set_aspect('equal')
ax.grid(True, linestyle='--')

for node_id, pos in noder.items():
    ax.plot(pos[0], pos[1], 'ks', markersize=8)
    ax.text(pos[0] - 0.5, pos[1] + 0.5, node_id, fontsize=12, fontweight='bold')

# Robot 1 (Blå)
safe_circle = patches.Circle((0, 0), safe_radius, fill=False, linestyle='--', color='blue', alpha=0.5)
ax.add_patch(safe_circle)
rect = patches.Rectangle((0, 0), L, W, color='blue', alpha=0.8, label='Robot 1')
ax.add_patch(rect)
trail, = ax.plot([], [], 'b-', linewidth=2)
ghost_dot, = ax.plot([], [], 'kx', markersize=5, alpha=0.9, label="Robot 1 referens")

# Robot 2 (Röd)
r2_safe_circle = patches.Circle((0, 0), safe_radius, fill=False, linestyle='--', color='red', alpha=0.5)
ax.add_patch(r2_safe_circle)
r2_rect = patches.Rectangle((0, 0), L, W, color='red', alpha=0.8, label='Robot 2')
ax.add_patch(r2_rect)
r2_trail, = ax.plot([], [], 'r-', linewidth=2)
r2_ghost_dot, = ax.plot([], [], 'kx', markersize=5, alpha=0.9, label="Robot 2 referens")

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_title("Två robotar med MPC + CBF")

prev_X_sol = None
prev_U_sol = None
last_u = np.array([0.0, 0.0])

prev_X_sol_r2 = None
prev_U_sol_r2 = None
last_u_r2 = np.array([0.0, 0.0])


def robot_dynamics(st, a, delta):
    dx = st[3] * np.cos(st[2])
    dy = st[3] * np.sin(st[2])
    dpsi = (st[3] / L) * np.tan(delta)
    dv = a
    return np.array([dx, dy, dpsi, dv])


def update(frame):
    global state, state_r2
    global prev_X_sol, prev_U_sol, last_u
    global prev_X_sol_r2, prev_U_sol_r2, last_u_r2

    current_time = frame * Ts

    ref_matrix = np.zeros((4, N + 1))
    ref_matrix_r2 = np.zeros((4, N + 1))

    for k in range(N + 1):
        t_future = current_time + k * Ts
        ref_matrix[:, k] = get_ref_at_time(t_future)
        ref_matrix_r2[:, k] = get_ref_at_time_r2(t_future)

    ghost_steps = 5
    ghost_ref_1 = ref_matrix[:, min(ghost_steps, N)]
    ghost_ref_2 = ref_matrix_r2[:, min(ghost_steps, N)]
    # MPC för robot 2
    opti_mpc_r2.set_value(X0_r2, state_r2)
    opti_mpc_r2.set_value(Xref_r2, ref_matrix_r2)

    if prev_X_sol_r2 is not None:
        opti_mpc_r2.set_initial(X_r2, prev_X_sol_r2)
    if prev_U_sol_r2 is not None:
        opti_mpc_r2.set_initial(U_r2, prev_U_sol_r2)

    try:
        sol_mpc_r2 = opti_mpc_r2.solve()
        u_r2_val = sol_mpc_r2.value(U_r2[:, 0])
        prev_X_sol_r2 = sol_mpc_r2.value(X_r2)
        prev_U_sol_r2 = sol_mpc_r2.value(U_r2)
    except RuntimeError:
        u_r2_val = last_u_r2.copy()

    r2_pos = state_r2[:2]
    r2_vel = np.array([
        state_r2[3] * np.cos(state_r2[2]),
        state_r2[3] * np.sin(state_r2[2])
    ])

    # Lös MPC för robot 1
    opti_mpc.set_value(X0, state)
    opti_mpc.set_value(Xref, ref_matrix)
    opti_mpc.set_value(obs_pos_mpc, r2_pos)
    opti_mpc.set_value(obs_vel_mpc, r2_vel)

    if prev_X_sol is not None:
        opti_mpc.set_initial(X, prev_X_sol)
    if prev_U_sol is not None:
        opti_mpc.set_initial(U, prev_U_sol)

    try:
        sol_mpc = opti_mpc.solve()
        u_mpc_val = sol_mpc.value(U[:, 0])
        prev_X_sol = sol_mpc.value(X)
        prev_U_sol = sol_mpc.value(U)
    except RuntimeError:
        u_mpc_val = opti_mpc.debug.value(U[:, 0])

    # Lös QP för robot 1
    opti_qp.set_value(X_curr, state)
    opti_qp.set_value(U_mpc_in, u_mpc_val)
    opti_qp.set_value(obs_pos_qp, r2_pos)
    opti_qp.set_value(obs_vel_qp, r2_vel)

    try:
        sol_qp = opti_qp.solve()
        u_star_val = sol_qp.value(U_star)
        last_u = u_star_val.copy()
    except RuntimeError:
        u_star_val = last_u.copy()

    control_hist.append(u_star_val.copy())
    r2_control_hist.append(u_r2_val.copy())

    # Uppdatera robot 1
    a1, delta1 = u_star_val[0], u_star_val[1]
    st1 = state.copy()

    k1 = robot_dynamics(st1, a1, delta1)
    k2 = robot_dynamics(st1 + Ts / 2 * k1, a1, delta1)
    k3 = robot_dynamics(st1 + Ts / 2 * k2, a1, delta1)
    k4 = robot_dynamics(st1 + Ts * k3, a1, delta1)

    state[:] = st1 + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Uppdatera robot 2
    a2, delta2 = u_r2_val[0], u_r2_val[1]
    st2 = state_r2.copy()

    k1 = robot_dynamics(st2, a2, delta2)
    k2 = robot_dynamics(st2 + Ts / 2 * k1, a2, delta2)
    k3 = robot_dynamics(st2 + Ts / 2 * k2, a2, delta2)
    k4 = robot_dynamics(st2 + Ts * k3, a2, delta2)

    state_r2[:] = st2 + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    last_u_r2 = u_r2_val.copy()

    goal_pos = noder["v15"]
    if np.linalg.norm(state[:2] - goal_pos) < 0.2:
        state[0] = goal_pos[0]
        state[1] = goal_pos[1]
        state[2] = np.arctan2(goal_pos[1] - noder["v0"][1], goal_pos[0] - noder["v0"][0])
        state[3] = 0.0

    goal_pos_r2 = noder["v0"]
    if np.linalg.norm(state_r2[:2] - goal_pos_r2) < 0.2:
        state_r2[0] = goal_pos_r2[0]
        state_r2[1] = goal_pos_r2[1]
        state_r2[2] = np.arctan2(goal_pos_r2[1] - noder["v15"][1], goal_pos_r2[0] - noder["v15"][0])
        state_r2[3] = 0.0

    # Stoppa animation när båda är framme
    if np.linalg.norm(state[:2] - noder["v15"]) < 0.2 and np.linalg.norm(state_r2[:2] - noder["v0"]) < 0.2:
        ani.event_source.stop()

    # Spara robot 1
    state_hist.append(state.copy())
    traj_x.append(state[0])
    traj_y.append(state[1])

    # Spara robot 2
    r2_state_hist.append(state_r2.copy())
    r2_traj_x.append(state_r2[0])
    r2_traj_y.append(state_r2[1])

    # RITA ROBOT 1
    x, y, psi, v = state
    diag = np.sqrt(L ** 2 + W ** 2) / 2
    q = np.arctan(W / L) + psi
    rect.set_xy([x - diag * np.cos(q), y - diag * np.sin(q)])
    rect.angle = np.degrees(psi)
    safe_circle.center = (x, y)
    trail.set_data(traj_x, traj_y)

    ghost_dot.set_data([ghost_ref_1[0]], [ghost_ref_1[1]])

    # RITA ROBOT 2
    r2_x, r2_y, r2_psi, r2_v = state_r2
    r2_q = np.arctan(W / L) + r2_psi
    r2_rect.set_xy([r2_x - diag * np.cos(r2_q), r2_y - diag * np.sin(r2_q)])
    r2_rect.angle = np.degrees(r2_psi)
    r2_safe_circle.center = (r2_x, r2_y)
    r2_trail.set_data(r2_traj_x, r2_traj_y)

    r2_ghost_dot.set_data([ghost_ref_2[0]], [ghost_ref_2[1]])
    time_text.set_text(f'Tid: {current_time:.1f}s')

    return (
        rect, safe_circle, trail, ghost_dot,
        r2_rect, r2_safe_circle, r2_trail, r2_ghost_dot,
        time_text
    )



ani = FuncAnimation(fig, update, frames=200, interval=Ts * 1000, blit=True, repeat=False)
plt.legend(loc="lower right")
plt.show()

# 6. FINAL PLOTTAR
s1 = np.array(state_hist)
s2 = np.array(r2_state_hist)
u1 = np.array(control_hist)
u2 = np.array(r2_control_hist)

fig_states, ax_states = plt.subplots(4, 1, sharex=True)
fig_ctrls, ax_ctrls = plt.subplots(2, 1, sharex=True)

labels_states = ['x', 'y', 'psi', 'v']
labels_ctrls = ['a', 'delta']

t_s1 = np.arange(len(s1)) * Ts
t_s2 = np.arange(len(s2)) * Ts
t_u1 = np.arange(len(u1)) * Ts
t_u2 = np.arange(len(u2)) * Ts

for i in range(4):
    ax_states[i].plot(t_s1, s1[:, i], 'b', label='Robot 1')
    ax_states[i].plot(t_s2, s2[:, i], 'r--', label='Robot 2')
    ax_states[i].set_ylabel(labels_states[i])
    ax_states[i].grid()

for i in range(2):
    ax_ctrls[i].plot(t_u1, u1[:, i], 'b', label='Robot 1')
    ax_ctrls[i].plot(t_u2, u2[:, i], 'r--', label='Robot 2')
    ax_ctrls[i].set_ylabel(labels_ctrls[i])
    ax_ctrls[i].grid()

ax_states[-1].set_xlabel("time [s]")
ax_ctrls[-1].set_xlabel("time [s]")

ax_states[0].legend()
ax_ctrls[0].legend()

plt.show()