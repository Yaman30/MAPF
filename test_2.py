import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation
import casadi as ca
from ReferenceTrajectoryUtils import get_instance_trajectories

############# EXEMPEL: hämta ref-trajectorier från fil #############
instance = 'instance_25254ef2-0876-4627-80d9-0c97b76cfbe9.json'

trajectories = get_instance_trajectories(instance, wait_dir='incoming')
# wait_dir = 'incoming': agentens ref-vinkel när den väntar är samma som den hade när den kom till noden
# wait_dir = 'outgoing': agentens ref-vinkel när den väntar är samma som den har när den lämnar noden (dvs riktning mot nästa nod)
# wait_dir = 'avg': agentens ref-vinkel när den väntar är genomsnittet av ingående och utgående riktning (om båda finns, annars samma som den riktning som finns)

# Trajectories (dict) contains one trajectory for each agent
agent_0_trajectory = trajectories['a0']
print(agent_0_trajectory(0))        # trajectory point at time t=0
print(agent_0_trajectory(56.7))     # trajectory point at time t=56.7
print(agent_0_trajectory(100))      # trajectory point at time t=100
##############################################################################


# 1. BANA OCH WAYPOINTS

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


class Robot:
    def __init__(self, id_num, start_state, waypoints, color, label):
        self.id = id_num
        # States: x, y, psi (vinkel), v (hastighet)
        self.state = np.array([*start_state, 0.0], dtype=float)
        self.color = color
        self.label = label
        self.s_fine, self.x_r, self.y_r, self.psi_r, self.delta_r, self.v_r = \
            skapa_bana(waypoints, L, v_max, ay_max)
        self.x_hist = [self.state[0]]
        self.y_hist = [self.state[1]]
        self.finished = False



# 2. INSTÄLLNINGAR

L = 1.0  # Robotens axelavstånd
W = 0.5  # Robotens bredd
Ts = 0.1  # Tidssteg i simuleringen

# Robotfysikens gränser
v_max = 2.0
ay_max = 0.6
a_max = 0.5
a_brake = 1.0
delta_max = 0.6  # Max styrvinkel i radianer


Look_ahead = 1.5  # Avståndet till referenspunkten och säkerhetszonens centrum
Safe_dist = 2.0  # Säkerhetszonens radie

# MPC Inställningar
N_horizon = 15  # Hur många steg MPC tittar in i framtiden

waypoints1 = np.array([[0, 2], [15, 2], [25, 2], [40, 12], [55, 12], [70, 12], [85, 2], [95, 2], [110, 2]])
waypoints2 = np.array([[0, 12], [15, 12], [25, 12], [40, 2], [55, 2], [70, 2], [85, 12], [95, 12], [110, 12]])

# Starttillstånd: [x, y, psi].
r1 = Robot(1, [0, 2, 0], waypoints1, 'cyan', 'Robot 1')
r2 = Robot(2, [0, 12, 0], waypoints2, 'orange', 'Robot 2')
robots = [r1, r2]


# 3. CASADI MPC HJÄRNA MED RK4
def create_mpc_solver():
    opti = ca.Opti()

    # Variabler som optimeraren får ändra på
    X = opti.variable(4, N_horizon + 1)  # States över tid: [x, y, psi, v]
    U = opti.variable(2, N_horizon)  # Controls över tid: [a, delta]

    # Nuvarande tillstånd och målbana
    X0 = opti.parameter(4)
    X_ref = opti.parameter(4, N_horizon + 1)

    opti.subject_to(X[:, 0] == X0)  # Roboten börjar där den är just nu

    cost = 0

    # Skapa symbolisk funktion för bilens rörelse
    state_sym = ca.SX.sym('x', 4)  # [x, y, psi, v]
    ctrl_sym = ca.SX.sym('u', 2)  # [a, delta]
    v_s = state_sym[3]
    psi_s = state_sym[2]
    a_s = ctrl_sym[0]
    delta_s = ctrl_sym[1]

    # Differentialekvationerna
    dx = v_s * ca.cos(psi_s)
    dy = v_s * ca.sin(psi_s)
    dpsi = (v_s / L) * ca.tan(delta_s)
    dv = a_s
    f_kin = ca.Function('f_kin', [state_sym, ctrl_sym], [ca.vertcat(dx, dy, dpsi, dv)])

    for k in range(N_horizon):
        # RK4 INTEGRATION I FRAMTIDEN

        st = X[:, k]
        con = U[:, k]
        k1 = f_kin(st, con)
        k2 = f_kin(st + Ts / 2 * k1, con)
        k3 = f_kin(st + Ts / 2 * k2, con)
        k4 = f_kin(st + Ts * k3, con)
        x_next_rk4 = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Tvinga systemet att följa fysikens lagar
        opti.subject_to(X[:, k + 1] == x_next_rk4)

        # Straffa fel och stora styrsignaler
        cost += 15.0 * (X[0, k] - X_ref[0, k]) ** 2  # x fel
        cost += 10.0 * (X[1, k] - X_ref[1, k]) ** 2  # y fel
        cost += 5.0 * (1 - ca.cos(X[2, k] - X_ref[2, k]))  # Vinkelfel
        cost += 2.0 * (X[3, k] - X_ref[3, k]) ** 2  # hastighetsfel

        cost += 0.5 * U[0, k] ** 2  # Straffa hård gas/broms
        cost += 5.0 * U[1, k] ** 2  # Straffa tvära rattrörelser

    # Begränsningar
    opti.subject_to(opti.bounded(-a_brake, U[0, :], a_max))
    opti.subject_to(opti.bounded(-delta_max, U[1, :], delta_max))
    opti.subject_to(opti.bounded(0, X[3, :], v_max))  # Får inte backa

    opti.minimize(cost)

    # Stäng av all text utskrift från Ipopt för att inte spamma terminalen
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)

    return opti, X0, X_ref, U


# Skapa en CasADi solver per robot
mpc_solvers = {r: create_mpc_solver() for r in robots}


# 4. GRAFIK
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(-2, 115)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("MPC CasADi Multi-Robot med Säkerhetszon och Referenspunkt")

# Rita referensbanorna
ax.plot(r1.x_r, r1.y_r, 'g--', alpha=0.3)
ax.plot(r2.x_r, r2.y_r, 'r--', alpha=0.3)

gfx_objects = {}
for r in robots:
    # Själva roboten rektangeln
    rect = patches.Rectangle((0, 0), L, W, color=r.color, alpha=0.8, label=r.label)
    ax.add_patch(rect)

    # Historik spåret
    trail, = ax.plot([], [], color=r.color, linewidth=2)

    # Säkerhetszonen
    target_circle = patches.Circle((0, 0), radius=Safe_dist/2, color='red', fill=False, linestyle='--', alpha=0.5,
                                   linewidth=1.5)
    ax.add_patch(target_circle)

    # Referenspunkten
    target_dot, = ax.plot([], [], 'ro', markersize=4, alpha=0.8)

    gfx_objects[r] = {'rect': rect, 'trail': trail, 'target_circle': target_circle, 'target_dot': target_dot}

status_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, color='red', fontweight='bold')
# Håller koll på krockar
krock_counter = 0
is_colliding = False


def update(frame):
    global krock_counter, is_colliding
    artists = [status_text]
    collision_this_frame = False

    # STATISTIK
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            ra, rb = robots[i], robots[j]
            if ra.finished or rb.finished: continue
            dist = np.sqrt((ra.state[0] - rb.state[0]) ** 2 + (ra.state[1] - rb.state[1]) ** 2)
            if dist < Safe_dist + 0.5:  # Lägger till lite marginal för krock-logiken
                collision_this_frame = True

    if collision_this_frame:
        if not is_colliding:
            krock_counter += 1
            is_colliding = True
    else:
        is_colliding = False

    status_text.set_text(f"Antal krockrisker: {krock_counter}")

    # STYRNING (MPC) & FYSIK (RK4)
    for r in robots:
        if not r.finished:
            x, y, psi, v = r.state
            goal_x, goal_y = r.x_r[-1], r.y_r[-1]
            dist_to_goal = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)

            if dist_to_goal < 0.5:
                r.finished = True
                r.state[:] = [goal_x, goal_y, psi, 0.0]
                gfx_objects[r]['target_circle'].set_alpha(0)
                gfx_objects[r]['target_dot'].set_alpha(0)
            else:
                # 1 Hitta var på referensbanan vi är för att plocka ut målpunkter för framtiden
                dists = (x - r.x_r) ** 2 + (y - r.y_r) ** 2
                nearest_idx = np.argmin(dists)

                # Bygg referensmatris (X_ref) för MPC-horisonten.
                ref_matrix = np.zeros((4, N_horizon + 1))
                step_size_horiz = max(1, int((max(v, 0.5) * Ts * 1.5) / (r.s_fine[1] - r.s_fine[0])))

                for k in range(N_horizon + 1):
                    # Plockar målpunkter från banan stegvis in i framtiden
                    idx = min(nearest_idx + k * step_size_horiz, len(r.x_r) - 1)
                    # För att stanna i slutet, tvinga hastigheten till 0 nära slutet
                    v_target = r.v_r[idx] if idx < len(r.x_r) - 100 else 0.0
                    ref_matrix[:, k] = [r.x_r[idx], r.y_r[idx], r.psi_r[idx], v_target]

                # 2 FRÅGA HJÄRNAN (CasADi Optimering
                opti, X0_p, X_ref_p, U_v = mpc_solvers[r]
                opti.set_value(X0_p, r.state)  # Skickar in nuvarande tillstånd
                opti.set_value(X_ref_p, ref_matrix)  # Skickar in referensmatrisen

                try:
                    sol = opti.solve()
                    u_opt = sol.value(U_v[:, 0])  # Ta bara först steget av den uträknade sekvensen [a, delta]
                except RuntimeError:
                    # Om optimeraren misslyckas
                    u_opt = opti.debug.value(U_v[:, 0])

                a_cmd, delta_cmd = u_opt[0], u_opt[1]

                # 3 RK4 Fysikmotor
                def robot_dynamics(st, a, delta):
                    # Definierar differentialekvationerna för rörelsen
                    dx = st[3] * np.cos(st[2])
                    dy = st[3] * np.sin(st[2])
                    dpsi = (st[3] / L) * np.tan(delta)
                    dv = a
                    return np.array([dx, dy, dpsi, dv])

                st = r.state
                # RK4: Beräknar 4 lutningar
                k1 = robot_dynamics(st, a_cmd, delta_cmd)
                k2 = robot_dynamics(st + Ts / 2 * k1, a_cmd, delta_cmd)
                k3 = robot_dynamics(st + Ts / 2 * k2, a_cmd, delta_cmd)
                k4 = robot_dynamics(st + Ts * k3, a_cmd, delta_cmd)

                # Uppdatera tillstånd exakt genom integration
                r.state = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

                r.x_hist.append(r.state[0])
                r.y_hist.append(r.state[1])

        # 4 RITA UPP (Positionering av grafik)
        x, y, psi, v = r.state
        goal_s = r.s_fine[-1]

        # Robot rektangeln
        diag = np.sqrt(L ** 2 + W ** 2) / 2
        q = np.arctan(W / L) + psi
        gfx_objects[r]['rect'].set_xy([x - diag * np.cos(q), y - diag * np.sin(q)])
        gfx_objects[r]['rect'].angle = np.degrees(psi)

        # Historik spåret
        gfx_objects[r]['trail'].set_data(r.x_hist, r.y_hist)

        if not r.finished:
            # Hitta var på banan vi är nu
            dists = (x - r.x_r) ** 2 + (y - r.y_r) ** 2
            nearest_idx = np.argmin(dists)

            # Uppdatera Referenspunkten
            step_lookahead = int(Look_ahead / (r.s_fine[1] - r.s_fine[0]))
            target_P_idx = min(nearest_idx + step_lookahead, len(r.x_r) - 1)
            target_x = r.x_r[target_P_idx]
            target_y = r.y_r[target_P_idx]
            gfx_objects[r]['target_dot'].set_data([target_x], [target_y])

            # Uppdatera Säkerhetszonen

            future_x = r.x_r[target_P_idx]
            future_y = r.y_r[target_P_idx]
            gfx_objects[r]['target_circle'].center = (future_x, future_y)

        artists.extend([gfx_objects[r]['rect'], gfx_objects[r]['trail'], gfx_objects[r]['target_circle'],
                        gfx_objects[r]['target_dot']])

    return artists


ani = FuncAnimation(fig, update, frames=1800, interval=Ts * 1000, blit=True)
plt.legend(loc='upper right')
ani.save('mpc_robot_med_punkt_och_zon.mp4', writer='ffmpeg', fps=30, dpi=100)
#plt.show()