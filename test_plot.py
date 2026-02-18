import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_discrete_are
from matplotlib.animation import FuncAnimation


# Skapa bana

def skapa_bana(waypoints, L, v_max, ay_max):
    dx = np.diff(waypoints[:, 0])
    dy = np.diff(waypoints[:, 1])
    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.concatenate(([0], np.cumsum(ds)))

    # Mjuka kurvor
    cs_x = CubicSpline(s, waypoints[:, 0], bc_type='natural')
    cs_y = CubicSpline(s, waypoints[:, 1], bc_type='natural')

    s_fine = np.linspace(0, s[-1], 2000)
    x_r = cs_x(s_fine)
    y_r = cs_y(s_fine)

    dx_ds = cs_x.derivative()(s_fine)
    dy_ds = cs_y.derivative()(s_fine)
    d2x_ds2 = cs_x.derivative(2)(s_fine)
    d2y_ds2 = cs_y.derivative(2)(s_fine)

    psi_r = np.arctan2(dy_ds, dx_ds)
    kappa_r = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds ** 2 + dy_ds ** 2) ** 1.5
    delta_r = np.arctan(L * kappa_r)

    v_r = np.minimum(v_max, np.sqrt(ay_max / (np.abs(kappa_r) + 1e-6)))

    return s_fine, x_r, y_r, psi_r, delta_r, v_r



# Klass

class Robot:
    def __init__(self, id_num, start_state, waypoints, color, label):
        self.id = id_num
        self.state = np.array(start_state, dtype=float)  # x, y, psi
        self.color = color
        self.label = label

        self.s_fine, self.x_r, self.y_r, self.psi_r, self.delta_r, self.v_r = \
            skapa_bana(waypoints, L, v_max, ay_max)

        self.x_hist = [self.state[0]]
        self.y_hist = [self.state[1]]

        self.finished = False



# Inställningar

L = 1.0
W = 0.5
Ts = 0.1
v_max = 1.0
ay_max = 0.6
Look_ahead = 1.5
Safe_dist = 2.0

Q = np.diag([15.0, 10.0])
R = np.array([[1.0]])

# Bana 1
waypoints1 = np.array([
    [0, 0], [5, 0], [10, 7], [13, 5],
    [16, 5], [20, 8], [25, -3]
])

# Bana 2
waypoints2 = np.array([
    [0, 7], [5, 7], [10, 0], [13, 2],
    [16, 2], [20, -1], [25, 9]
])
r1 = Robot(1,[0, 0, 0], waypoints1, 'cyan', 'Robot 1')
r2 = Robot(2,[0, 7, 0], waypoints2, 'orange', 'Robot 2')
robots = [r1, r2]


#LQR

def get_lqr_gain(v_nom):
    if v_nom < 0.1: v_nom = 0.1
    A = np.array([[1.0, v_nom * Ts], [0.0, 1.0]])
    B = np.array([[0.0], [(v_nom * Ts) / L]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi



# Grafik

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-2, 28)
ax.set_ylim(-5, 10)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("LQR Multi-Robot")

ax.plot(r1.x_r, r1.y_r, 'g--', alpha=0.3)
ax.plot(r2.x_r, r2.y_r, 'r--', alpha=0.3)

gfx_objects = {}
for r in robots:
    rect = patches.Rectangle((0, 0), L, W, color=r.color, alpha=0.8, label=r.label)
    ax.add_patch(rect)
    trail, = ax.plot([], [], color=r.color, linewidth=2)
    target_dot, = ax.plot([], [], 'ro', markersize=4, alpha=0.5)
    gfx_objects[r] = {'rect': rect, 'trail': trail, 'target': target_dot}

status_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, color='red', fontweight='bold')


def update(frame):
    #  Kollision

    artists = [status_text]
    move_permissions = {r: True for r in robots}
    warning = ""


    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            ra = robots[i]
            rb = robots[j]


            if ra.finished or rb.finished:
                continue


            dist = np.sqrt((ra.state[0] - rb.state[0]) ** 2 + (ra.state[1] - rb.state[1]) ** 2)

            if dist < Safe_dist:
                warning = "VARNING: Krockrisk!"

                if ra.id > rb.id:
                    move_permissions[ra] = False
                else:
                    move_permissions[rb] = False

    status_text.set_text(warning)


    for r in robots:
        if not r.finished:
            x, y, psi = r.state

            # Hitta var vi är på banan
            dists = (x - r.x_r) ** 2 + (y - r.y_r) ** 2
            nearest_idx = np.argmin(dists)
            current_s = r.s_fine[nearest_idx]

            # Hitta Målet
            goal_x = r.x_r[-1]
            goal_y = r.y_r[-1]

            # Beräkna avstånd till sista punkten
            dist_to_goal = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)

            # Stanna på målet
            if dist_to_goal < 0.1:
                r.finished = True
                r.state[0] = goal_x
                r.state[1] = goal_y

            else:
                target_s = current_s + Look_ahead
                target_idx = np.searchsorted(r.s_fine, target_s)
                if target_idx >= len(r.s_fine):
                    target_idx = len(r.s_fine) - 1

                xr, yr = r.x_r[target_idx], r.y_r[target_idx]
                psir, deltar = r.psi_r[target_idx], r.delta_r[target_idx]

                # Bromsa mjukt
                dist_remaining = r.s_fine[-1] - current_s
                vr = r.v_r[target_idx]

                if dist_remaining < 1.0:
                    vr = vr * (dist_remaining / 1.0)
                    if vr < 0.2: vr = 0.2

                if not move_permissions[r]:
                    vr = 0.0

                ey = -np.sin(psir) * (x - xr) + np.cos(psir) * (y - yr)
                epsi = wrap_angle(psi - psir)

                K = get_lqr_gain(vr)
                delta_delta = -(K @ np.array([ey, epsi]))[0]
                delta = np.clip(deltar + delta_delta, -0.6, 0.6)

                x += vr * np.cos(psi) * Ts
                y += vr * np.sin(psi) * Ts
                psi += (vr / L) * np.tan(delta) * Ts

                r.state[:] = [x, y, psi]
                r.x_hist.append(x)
                r.y_hist.append(y)

                gfx_objects[r]['target'].set_data([xr], [yr])

        # Rita
        x, y, psi = r.state

        diag = np.sqrt(L ** 2 + W ** 2) / 2
        q = np.arctan(W / L) + psi
        gfx_objects[r]['rect'].set_xy([x - diag * np.cos(q), y - diag * np.sin(q)])
        gfx_objects[r]['rect'].angle = np.degrees(psi)

        gfx_objects[r]['trail'].set_data(r.x_hist, r.y_hist)

        artists.extend([gfx_objects[r]['rect'], gfx_objects[r]['trail'], gfx_objects[r]['target']])


    return artists


ani = FuncAnimation(fig, update, frames=600, interval=30, blit=True)
plt.legend(loc='upper right')
plt.show()