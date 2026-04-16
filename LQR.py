import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_discrete_are

from ReferenceTrajectoryUtils import get_instance_trajectories

# 1. INSTÄLLNINGAR
L = 1.0
W = 0.5
Ts = 0.1
v_max = 2.0
delta_max = 1.0
COLLISION_RADIUS = 1.0
SAFE_ZONE_RADIUS = 0.8
LOOKAHEAD_TIME = 0.3
GHOST_VISUAL_TIME = 1.0
# LQR-vikter
Q = np.diag([15.0, 10.0])
R = np.array([[1.0]])

# Instansnamn
INSTANCE_NAME = 'instance_f229f7bd-94be-4fe8-890b-c827558e6d6e.json'

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unwrap_angle_near_reference(angle, reference):
    while angle - reference > np.pi:
        angle -= 2 * np.pi
    while angle - reference < -np.pi:
        angle += 2 * np.pi
    return angle


def get_lqr_gain(v_nom):
    if v_nom < 0.1:
        v_nom = 0.1

    A = np.array([
        [1.0, v_nom * Ts],
        [0.0, 1.0]
    ])

    B = np.array([
        [0.0],
        [(v_nom * Ts) / L]
    ])

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K


def build_reference_state(robot):
    t_ref = robot.progress_time + LOOKAHEAD_TIME
    t_ref2 = t_ref + Ts
    t_ghost = robot.progress_time + GHOST_VISUAL_TIME

    ref1 = robot.trajectory(t_ref)
    ref2 = robot.trajectory(t_ref2)
    ghost = robot.trajectory(t_ghost)

    xr = float(ref1[0])
    yr = float(ref1[1])
    psir_raw = float(ref1[2])
    vr = float(ref1[3])

    #xr2 = float(ref2[0])
    #yr2 = float(ref2[1])
    psir2_raw = float(ref2[2])
    vr2 = float(ref2[3])

    current_psi = robot.state[2]
    if vr < 0.05:
        psir = current_psi
    else:
        psir = unwrap_angle_near_reference(psir_raw, current_psi)

    if vr2 < 0.05:
        psir2 = psir
    else:
        psir2 = unwrap_angle_near_reference(psir2_raw, psir)

    psidot_ref = wrap_angle(psir2 - psir) / Ts
    deltar = np.arctan2(L * psidot_ref, max(vr, 0.1))

    deltar = np.clip(deltar, -delta_max, delta_max)
    vr = np.clip(vr, 0.0, v_max)

    ghost_xy = [float(ghost[0]), float(ghost[1])]

    return xr, yr, psir, vr, deltar, ghost_xy


def robot_step(state, v_cmd, delta_cmd):

    x, y, psi = state

    x += v_cmd * np.cos(psi) * Ts
    y += v_cmd * np.sin(psi) * Ts
    psi += (v_cmd / L) * np.tan(delta_cmd) * Ts
    psi = wrap_angle(psi)

    return np.array([x, y, psi], dtype=float)

# 3. ROBOTKLASS
class Robot:
    def __init__(self, id_num, agent_id, trajectory_fn, color, label):
        self.progress_time = 0.0
        self.id = id_num
        self.agent_id = agent_id
        self.trajectory = trajectory_fn
        self.color = color
        self.label = label

        start_ref = self.trajectory(0.0)
        self.state = np.array([(start_ref[0]), (start_ref[1]), (start_ref[2])], dtype=float)

        self.start_xy = np.array([float(start_ref[0]), float(start_ref[1])], dtype=float)

        final_ref = self.trajectory(10000.0)
        self.goal_xy = np.array([float(final_ref[0]), float(final_ref[1])], dtype=float)
        self.goal_psi = float(final_ref[2])

        self.x_hist = [self.state[0]]
        self.y_hist = [self.state[1]]

        # För statistik
        self.state_hist = []     # [x, y, psi, v_ref]
        self.control_hist = []   # [delta]
        self.error_hist = []     # [ey, epsi]

        self.finished = False

    def get_final_goal(self):
        return self.goal_xy[0], self.goal_xy[1], self.goal_psi

# 4. LÄS IN INSTANS OCH SKAPA ROBOTAR

trajectories = get_instance_trajectories(INSTANCE_NAME, wait_dir='incoming')
robots = []
colors = ['cyan', 'orange', 'green', 'purple', 'red', 'blue', 'pink', 'brown', 'gray', 'olive']

for i, (agent_id, traj_fn) in enumerate(trajectories.items()):
    color = colors[i % len(colors)]
    robots.append(Robot(i + 1, agent_id, traj_fn, color, f"Robot {agent_id}"))

# 5. GRAFIK OCH ANIMERING

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_xlim(-2, 85)
ax.set_ylim(-2, 50)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("LQR + JSON")

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
    target_marker, = ax.plot([], [], marker='o', color=r.color, markersize=5, alpha=0.6,  linestyle='None')

    ax.add_patch(rect)
    ax.add_patch(safe_zone)

    gfx_objects[r.id] = {
        'rect': rect, 'trail': trail, 'safe_zone': safe_zone,
        'start_marker': start_marker, 'goal_marker': goal_marker, 'ghost_marker': ghost_marker, 'target_marker': target_marker
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

    # Kollisionsräkning
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

    # Uppdatera robotar
    for r in robots:
        gfx = gfx_objects[r.id]

        if not r.finished:
            x, y, psi = r.state

            xr, yr, psir, vr, deltar, ghost_xy = build_reference_state(r)

            # Ghost + target
            gfx['ghost_marker'].set_data([ghost_xy[0]], [ghost_xy[1]])
            gfx['target_marker'].set_data([xr], [yr])

            # LQR-fel
            ey = -np.sin(psir) * (x - xr) + np.cos(psir) * (y - yr)
            epsi = wrap_angle(psi - psir)

            K = get_lqr_gain(vr)
            delta_delta = -(K @ np.array([ey, epsi]))[0]
            delta = np.clip(deltar + delta_delta, -delta_max, delta_max)

            # Uppdatera tillstånd med referenshastigheten
            #r.state = robot_step(r.state, vr, delta)
            pos_error = np.hypot(x - xr, y - yr)
            v_cmd = np.clip(vr - 0.4 * pos_error, 0.0, v_max)
            r.state = robot_step(r.state, v_cmd, delta)

            if pos_error < 0.5:
                robot_progress_speed = 1.0
            elif pos_error < 1.5:
                robot_progress_speed = 0.4
            else:
                robot_progress_speed = 0.1

            r.progress_time += Ts * robot_progress_speed
            # Historik
            r.x_hist.append(r.state[0])
            r.y_hist.append(r.state[1])
            r.state_hist.append([r.state[0], r.state[1], r.state[2], vr])
            r.control_hist.append([delta])
            r.error_hist.append([ey, epsi])

            # Målkriterium
            goal_x, goal_y, goal_psi = r.get_final_goal()
            goal_dist = np.hypot(r.state[0] - goal_x, r.state[1] - goal_y)


            if goal_dist < 0.6 and vr < 0.1:
                r.finished = True
                r.state[:] = [goal_x, goal_y, goal_psi]
                gfx['ghost_marker'].set_data([], [])
                gfx['target_marker'].set_data([], [])
        else:
            gfx['ghost_marker'].set_data([], [])
            gfx['target_marker'].set_data([], [])

        # Rita robot
        x, y, psi = r.state

        diag = np.sqrt(L ** 2 + W ** 2) / 2
        q = np.arctan(W / L) + psi

        gfx['rect'].set_xy([x - diag * np.cos(q), y - diag * np.sin(q)])
        gfx['rect'].angle = np.degrees(psi)

        gfx['trail'].set_data(r.x_hist, r.y_hist)
        gfx['safe_zone'].center = (x, y)

        artists.extend([
            gfx['rect'],
            gfx['trail'],
            gfx['safe_zone'],
            gfx['start_marker'],
            gfx['goal_marker'],
            gfx['ghost_marker'],
            gfx['target_marker']
        ])

    return artists

ani = FuncAnimation(fig, update, frames=2000, interval=Ts * 1000, blit=False)
plt.legend(loc='upper right')
ani.save('lqr_robot_ghost_reference.mp4', writer='ffmpeg', fps=30, dpi=100)
plt.show()

# 6. STATES / CONTROLS / ERRORS / STATISTIK

fig_states, ax_states = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
fig_ctrls, ax_ctrls = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
fig_errs, ax_errs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

labels_states = ['x', 'y', 'psi', 'v_ref']
labels_errs = ['e_y', 'e_psi']

for r in robots:
    s = np.array(r.state_hist)
    u = np.array(r.control_hist)
    e = np.array(r.error_hist)

    if len(s) > 0:
        t_s = np.arange(len(s)) * Ts
        for i in range(4):
            ax_states[i].plot(t_s, s[:, i], label=r.label)
            ax_states[i].set_ylabel(labels_states[i])
            ax_states[i].grid(True)

    if len(u) > 0:
        t_u = np.arange(len(u)) * Ts
        ax_ctrls.plot(t_u, u[:, 0], label=r.label)
        ax_ctrls.set_ylabel('delta')
        ax_ctrls.grid(True)

    if len(e) > 0:
        t_e = np.arange(len(e)) * Ts
        for i in range(2):
            ax_errs[i].plot(t_e, e[:, i], label=r.label)
            ax_errs[i].set_ylabel(labels_errs[i])
            ax_errs[i].grid(True)

ax_states[-1].set_xlabel("time (s)")
ax_ctrls.set_xlabel("time (s)")
ax_errs[-1].set_xlabel("time (s)")

handles, labels = ax_states[0].get_legend_handles_labels()
if labels:
    ax_states[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax_ctrls.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax_errs[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1))

plt.show()