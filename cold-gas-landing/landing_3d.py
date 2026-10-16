"""3D rigid-body landing

 - Four ideal, parallel cold-gas thrusters control descent, roll and pitch.
They cannot directly produce yaw torque. Horizontal position is uncontrolled.
No gas depletion, drag, wind, sensor errors, suspension or ground collision.
z=0 is a centre-of-mass touchdown reference, not a physical contact model.
All parameters are illustrative, not Tesla specifications.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


MASS, GRAVITY = 10.0, 9.81
ARM, MAX_THRUST = 0.5, 40.0
INERTIA = np.array([2.0, 2.0, 3.0])  # Principal moments, kg m^2.
MOUNTS = np.array([[ARM, ARM, 0], [-ARM, ARM, 0],
                   [-ARM, -ARM, 0], [ARM, -ARM, 0]])


def rotation(vector):
    """Rotation matrix from an axis times its angle, in radians."""
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def simulate(dt=0.005):
    position = np.array([0.0, 0.0, 3.0])
    velocity = np.zeros(3)
    omega = np.zeros(3)  # Angular velocity in body coordinates.
    # Body-to-world orientation: start with both roll and pitch tilted.
    attitude = rotation(np.radians([0, -12, 0])) @ rotation(np.radians([10, 0, 0]))
    history = []
    for step in range(int(20 / dt)):
        desired_vz = -min(0.7, max(0.15, 0.8 * position[2]))
        desired_az = 3.0 * (desired_vz - velocity[2])
        total = MASS * (GRAVITY + desired_az) / max(attitude[2, 2], 0.2)

        # Express world-up in the body frame and rotate body-up towards it.
        target_up = attitude.T @ np.array([0.0, 0.0, 1.0])
        error = np.cross([0.0, 0.0, 1.0], target_up)
        torque = INERTIA * (8.0 * error - 4.0 * omega)
        torque[2] = 0.0  # Parallel jets cannot control yaw.

        # For each jet, r x F = [y*F, -x*F, 0].
        forces = total / 4 + (MOUNTS[:, 1] * torque[0]
                              - MOUNTS[:, 0] * torque[1]) / (4 * ARM**2)
        forces = np.clip(forces, 0, MAX_THRUST)
        history.append((step * dt, position.copy(), velocity.copy(),
                        attitude.copy(), forces.copy()))
        actual_torque = np.array([MOUNTS[:, 1] @ forces,
                                  -MOUNTS[:, 0] @ forces, 0.0])
        acceleration = attitude[:, 2] * forces.sum() / MASS - [0, 0, GRAVITY]
        # Euler's rigid-body equation includes coupling between rotation axes.
        angular_acceleration = (actual_torque - np.cross(omega, INERTIA * omega)) / INERTIA
        velocity += acceleration * dt
        position += velocity * dt
        omega += angular_acceleration * dt
        attitude = attitude @ rotation(omega * dt)
        if position[2] <= 0:
            history.append(((step + 1) * dt, position.copy(), velocity.copy(),
                            attitude.copy(), forces.copy()))
            break
    return history


def render(history):
    path = np.array([row[1] for row in history])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    low = path.min(axis=0) - [1, 1, 0.5]
    high = path.max(axis=0) + [1, 1, 1]
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
           zlim=(low[2], high[2]), xlabel="X (m)", ylabel="Y (m)",
           zlabel="Touchdown clearance (m)")
    ax.set_box_aspect(high - low)  # Equal physical scale on all axes.
    ax.view_init(elev=24, azim=-55)
    gx, gy = np.meshgrid([low[0], high[0]], [low[1], high[1]])
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.12, color="gray")
    ax.plot(*path.T, "--", color="0.7", label="Full trajectory")
    ax.plot(path[:, 0], path[:, 1], np.zeros(len(path)), ":", color="0.6")
    trail, = ax.plot([], [], [], color="tab:blue", label="Travelled path")
    body, = ax.plot([], [], [], "o-", color="tab:orange", lw=3, label="Vehicle")
    heading, = ax.plot([], [], [], color="tab:red", lw=3, label="Body forward")
    arrows = [None]
    status = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("3D landing with four thrusters\n"
                 "Blue arrows: thrust on body | Horizontal drift and yaw uncontrolled", pad=28)

    def draw(index):
        t, pos, vel, attitude, forces = history[index]
        mounts = MOUNTS @ attitude.T + pos
        outline = mounts[[0, 1, 2, 3, 0]]
        body.set_data_3d(*outline.T)
        front = np.array([pos, pos + attitude[:, 0] * ARM])
        heading.set_data_3d(*front.T)
        trail.set_data_3d(*path[:index + 1].T)
        if arrows[0] is not None:
            arrows[0].remove()
        vectors = forces[:, None] / MAX_THRUST * 0.8 * attitude[:, 2]
        arrows[0] = ax.quiver(*mounts.T, *vectors.T, color="tab:blue",
                              arrow_length_ratio=0.25, linewidth=2)
        tilt = np.degrees(np.arccos(np.clip(attitude[2, 2], -1, 1)))
        status.set_text(f"t = {t:.2f} s | tilt from upright = {tilt:.1f} deg\n"
                        f"Vertical speed = {vel[2]:.2f} m/s\n"
                        f"Horizontal speed = {np.linalg.norm(vel[:2]):.2f} m/s")

    # 20 fps playback; integration remains 200 Hz.
    frames = list(range(0, len(history), 10))
    if frames[-1] != len(history) - 1:
        frames.append(len(history) - 1)
    animation = FuncAnimation(fig, draw, frames=frames, interval=50,
                              init_func=lambda: None)
    folder = Path(__file__).parent
    animation.save(folder / "landing_3d.gif", writer=PillowWriter(fps=20), dpi=85)
    draw(len(history) - 1)
    fig.savefig(folder / "trajectory_3d.png", dpi=140)
    plt.close(fig)
    print(f"Animation and trajectory saved in {folder.resolve()}")


if __name__ == "__main__":
    history = simulate()
    t, position, velocity, attitude, _ = history[-1]
    tilt = np.degrees(np.arccos(np.clip(attitude[2, 2], -1, 1)))
    print(f"{'Touchdown' if position[2] <= 0 else 'Timeout'} at {t:.2f} s")
    print(f"Vertical speed: {velocity[2]:.3f} m/s; tilt: {tilt:.3f} degrees")
    print(f"Horizontal speed: {np.linalg.norm(velocity[:2]):.3f} m/s")
    print(f"X/Y drift: {position[0]:.3f}, {position[1]:.3f} m")
    print("Impact is not simulated; a level body alone does not guarantee a safe landing.")
    render(history)
