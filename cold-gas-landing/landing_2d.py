"""2D landing model. Run: python cold-gas-landing/landing_2d.py

Saves diagnostic plots, a spatial trajectory, and an animated GIF beside this file.

Simple, ideal continuous thrusters
 - no tank depletion, wind, sensor errors or suspension.
The controller levels the body and slows descent; it does not control sideways drift.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save a figure without needing a desktop window.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch


def clamp(value, low, high):
    return max(low, min(value, high))


# Illustrative values, not Tesla specifications. All quantities use SI units.
mass, gravity = 10.0, 9.81
arm, inertia = 0.5, 2.0  # Thruster distance from centre; rotational inertia.
max_thrust = 80.0        # Limit per thruster, newtons.
dt = 0.005
x, height = 0.0, 3.0    # Height means centre-of-mass clearance to touchdown.
vx, vz = 0.0, 0.0
angle, spin = math.radians(10), 0.0
history = []
landed = False

for step in range(int(20 / dt)):
    # Request a gentle descent, slowing near the ground.
    desired_vz = -min(0.7, max(0.15, 0.8 * height))
    desired_az = 3.0 * (desired_vz - vz)
    total = mass * (gravity + desired_az) / max(math.cos(angle), 0.2)

    # A spring-like correction levels the body; damping stops its rotation.
    torque = inertia * (-8.0 * angle - 4.0 * spin)
    left = clamp(total / 2 - torque / (2 * arm), 0, max_thrust)
    right = clamp(total / 2 + torque / (2 * arm), 0, max_thrust)
    history.append((step * dt, height, vz, math.degrees(angle), x, left, right))

    # Newton's laws: forces change velocity; torque changes angular velocity.
    thrust = left + right
    ax = -thrust * math.sin(angle) / mass
    az = thrust * math.cos(angle) / mass - gravity
    angular_acceleration = arm * (right - left) / inertia
    vx += ax * dt
    vz += az * dt
    spin += angular_acceleration * dt
    x += vx * dt
    height += vz * dt
    angle += spin * dt

    if height <= 0:
        landed = True
        history.append(((step + 1) * dt, height, vz, math.degrees(angle), x, left, right))
        print(f"Touchdown at {(step + 1) * dt:.2f} s")
        print(f"Vertical speed: {vz:.3f} m/s; tilt: {math.degrees(angle):.3f} degrees")
        print(f"Sideways speed: {vx:.3f} m/s; drift: {x:.3f} m")
        print("Contact ends the simulation; impact and tipping are not modelled.")
        break

if not landed:
    print("No touchdown within 20 seconds.")

time, heights, speeds, angles, positions, lefts, rights = zip(*history)
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for ax, values, label in zip(axes.flat,
                           [heights, speeds, angles, positions],
                           ["Clearance (m)", "Vertical speed (m/s)",
                            "Tilt (degrees)", "Sideways drift (m)"]):
    ax.plot(time, values)
    ax.set_ylabel(label)
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.3)
fig.suptitle("Two-thruster demo: level the body and slow descent\n"
             "Sideways drift remains uncontrolled")
fig.tight_layout()
output = Path(__file__).with_name("landing.png")
fig.savefig(output, dpi=150)
plt.close(fig)
print(f"Plot saved to {output}")

# Spatial view: a schematic bar represents the body, with a thruster at each end.
# z=0 is the model's touchdown reference, not a wheel/ground contact simulation.
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(min(positions) - 1.2, max(positions) + 1.2)
ax.set_ylim(-0.8, max(heights) + 1.5)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Horizontal position (m)")
ax.set_ylabel("Clearance above touchdown reference (m)")
ax.set_title("2D landing: tilt correction and controlled descent\n"
             "Blue arrows = thrust on body (length proportional to force)")
ax.grid(alpha=0.2)
ax.plot(positions, heights, "--", color="0.75", label="Full trajectory")
trail, = ax.plot([], [], color="tab:blue", label="Travelled path")
body, = ax.plot([], [], "o-", color="tab:orange", linewidth=6,
                markersize=5, label="Vehicle (schematic)")
centre, = ax.plot([], [], "ko", markersize=4)
arrows = [FancyArrowPatch((0, 0), (0, 1), arrowstyle="-|>",
                         mutation_scale=16, color="tab:blue", linewidth=2)
          for _ in range(2)]
for arrow in arrows:
    ax.add_patch(arrow)
status = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                 bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
ax.legend(loc="lower left", fontsize=8)


def draw_frame(index):
    t, z, vertical_speed, tilt, horizontal, fl, fr = history[index]
    theta = math.radians(tilt)
    c, s = math.cos(theta), math.sin(theta)
    ends_x = [horizontal - arm * c, horizontal + arm * c]
    ends_z = [z - arm * s, z + arm * s]
    body.set_data(ends_x, ends_z)
    centre.set_data([horizontal], [z])
    trail.set_data(positions[:index + 1], heights[:index + 1])
    for arrow, px, pz, force in zip(arrows, ends_x, ends_z, [fl, fr]):
        length = force / max_thrust * 0.8
        arrow.set_positions((px, pz), (px - length * s, pz + length * c))
    status.set_text(f"t = {t:.2f} s    tilt = {tilt:.1f} deg\n"
                    f"Vertical speed = {vertical_speed:.2f} m/s\n"
                    f"Left thrust = {fl:.1f} N    Right thrust = {fr:.1f} N\n"
                    "Sideways drift is uncontrolled")
    return body, centre, trail, status, *arrows


# Keep the simulation at 200 Hz but export at 25 frames per second.
frames = list(range(0, len(history), 8))
if frames[-1] != len(history) - 1:
    frames.append(len(history) - 1)
fig.tight_layout()
animation = FuncAnimation(fig, draw_frame, frames=frames, interval=40, blit=False)
gif_path = Path(__file__).with_name("landing_2d.gif")
animation.save(gif_path, writer=PillowWriter(fps=25), dpi=90)
draw_frame(len(history) - 1)
trajectory_path = Path(__file__).with_name("trajectory_2d.png")
fig.savefig(trajectory_path, dpi=150)
plt.close(fig)
print(f"Animation saved to {gif_path}")
print(f"Trajectory saved to {trajectory_path}")
