
"""
    Orbit Graphic Generator for 2.5x Kerbin-scale Earth.
"""


from typing import Final
import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Constants of 2.5x Kerbal-scale Earth
GRAVITATIONAL_CONSTANT: Final[float] = 6.6743015e-11
MOLAR_GAS_CONSTANT: Final[float] = 8.31446261815324
STANDARD_GRAVITY: Final[float] = 9.80665

MASS_OF_EARTH: Final[float] = 3.7327282e23
RADIUS_OF_EARTH: Final[float] = 1_592_750

ATMOSPHERE_MAX_ALTITUDE_MSL: Final[float] = 95_000
ATMOSPHERE_MOLAR_MASS: Final[float] = 0.0289644
MEAN_ATMOSPHERIC_TEMPERATURE: Final[float] = 288
ATMOSPHERE_SCALE_HEIGHT: Final[float] = (
    (MOLAR_GAS_CONSTANT * MEAN_ATMOSPHERIC_TEMPERATURE) /
    (ATMOSPHERE_MOLAR_MASS * STANDARD_GRAVITY)
)
ATMOSPHERE_PRESSURE_SEA_LEVEL: Final[float] = 1.01325e5
ATMOSPHERE_TEMPERATURE_SEA_LEVEL: Final[float] = 288
ATMOSPHERE_DENSITY_SEA_LEVEL: Final[float] = (
    (ATMOSPHERE_PRESSURE_SEA_LEVEL * ATMOSPHERE_MOLAR_MASS) /
    (MOLAR_GAS_CONSTANT * ATMOSPHERE_TEMPERATURE_SEA_LEVEL)
)

CROSS_SECTIONAL_AREA: Final[float] = 0.181
VEHICLE_MASS: Final[float] = 1_258.0
DIMENSIONLESS_DRAG_COEFFICIENT_0: Final[float] = 0.4695


def equations(current_time: float, current_state: tuple[float, ...]) -> tuple[float, ...]:
    """
    Computes the derivatives of position and velocity in a gravitational field.
    """
    (
        x_position, y_position, z_position,
        x_velocity, y_velocity, z_velocity
    ) = current_state

    distance_from_center_of_earth = np.sqrt(
        (x_position ** 2) +
        (y_position ** 2) +
        (z_position ** 2)
    )

    # Acceleration due to gravity
    gravitational_acceleration: Final[tuple[float, ...]] = np.multiply(
        (
            -(GRAVITATIONAL_CONSTANT * MASS_OF_EARTH) /
            (distance_from_center_of_earth ** 3)
        ),
        (x_position, y_position, z_position)
    )

    x_acceleration, y_acceleration, z_acceleration = gravitational_acceleration

    altitude_msl: float = distance_from_center_of_earth - RADIUS_OF_EARTH

    # print(f'{altitude_msl=}')

    outside_of_atmosphere: bool = altitude_msl > ATMOSPHERE_MAX_ALTITUDE_MSL
    if outside_of_atmosphere:
        return (
            x_velocity, y_velocity, z_velocity,
            x_acceleration, y_acceleration, z_acceleration
        )

    # Atmospheric density (exponential decay model, adjust scale height as needed)
    # ρ = ρ0 * exp(-h / H), where H is the scale height and ρ0 is the surface density
    atmospheric_density = ATMOSPHERE_DENSITY_SEA_LEVEL * np.exp(-altitude_msl / ATMOSPHERE_SCALE_HEIGHT) if altitude_msl > 0 else 0

    # Calculate the velocity magnitude
    velocity_magnitude: float = np.sqrt(x_velocity**2 + y_velocity**2 + z_velocity**2)

    # Calculate drag force magnitude
    atmospheric_drag_force: float = 0.5 * atmospheric_density * (velocity_magnitude ** 2) * DIMENSIONLESS_DRAG_COEFFICIENT_0 * CROSS_SECTIONAL_AREA
    atmospheric_drag_acceleration: float = atmospheric_drag_force / VEHICLE_MASS

    velocity_direction = np.divide((x_velocity, y_velocity, z_velocity), velocity_magnitude)

    atmospheric_drag_acceleration_vector = np.multiply(atmospheric_drag_acceleration, -velocity_direction)

    # Total acceleration (gravity + drag)
    x_acceleration += atmospheric_drag_acceleration_vector[0]
    y_acceleration += atmospheric_drag_acceleration_vector[1]
    z_acceleration += atmospheric_drag_acceleration_vector[2]

    return (
        x_velocity, y_velocity, z_velocity,
        x_acceleration, y_acceleration, z_acceleration
    )


def altitude_event(current_time: float, current_state: tuple[float, ...]) -> float:
    """
    Event function to stop the simulation when the object reaches the ground (altitude = 0).
    """
    x_position, y_position, z_position, _, _, _ = current_state
    distance_from_center_of_earth = np.sqrt(x_position**2 + y_position**2 + z_position**2)
    altitude_msl = distance_from_center_of_earth - RADIUS_OF_EARTH
    return altitude_msl  # Will trigger when altitude_msl == 0

altitude_event.terminal = True  # Stop integration when the event is triggered
altitude_event.direction = -1   # Trigger only when altitude is decreasing


# Function to create Earth with texture
def create_textured_sphere(image_file: str) -> mlab.figure:
    """
    Creates a 3D textured sphere representing Earth.
    """
    # Create figure window
    figure = mlab.figure(bgcolor=(0, 0, 0), size=(600, 600))

    # Load and map the texture
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)

    # Create the sphere source with texture
    Nrad = 180
    sphere = tvtk.TexturedSphereSource(radius=RADIUS_OF_EARTH, theta_resolution=Nrad, phi_resolution=Nrad)
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    figure.scene.add_actor(sphere_actor)

    return figure


def main() -> None:
    """
    Runs the orbit simulation and displays the trajectory around a textured Earth.
    """

    x0_position: Final[float] = -(RADIUS_OF_EARTH + 100_000.0)
    y0_position: Final[float] = 0.0
    z0_position: Final[float] = 0.0
    x0_velocity: Final[float] = 0.0
    y0_velocity: Final[float] = -(3_830.0 - 94.0)
    z0_velocity: Final[float] = 0.0

    print(RADIUS_OF_EARTH + 100_000)

    initial_state: Final[tuple[float, ...]] = (
        x0_position,
        y0_position,
        z0_position,
        x0_velocity,
        y0_velocity,
        z0_velocity
    )

    time_span: Final[tuple[float, float]] = (0.0, 100_000.0)
    time_eval = np.linspace(time_span[0], time_span[1], 10_000_000)  # Time points for evaluation

    solution = solve_ivp(
        fun=equations,
        t_span=time_span,
        y0=initial_state,
        t_eval=time_eval,
        method='RK23',
        events=altitude_event  # Include the event function
    )

    #  Calculate altitude, velocity, and acceleration
    time_values = solution.t

    x_positions, y_positions, z_positions = solution.y[0], solution.y[1], solution.y[2]
    x_velocities, y_velocities, z_velocities = solution.y[3], solution.y[4], solution.y[5]

    distance_from_center: np.ndarray = np.sqrt(x_positions ** 2 + y_positions ** 2 + z_positions ** 2)
    altitudes_msl: np.ndarray = distance_from_center - RADIUS_OF_EARTH

    vertical_velocities: np.ndarray = np.gradient(altitudes_msl, time_values)
    velocity_magnitudes: np.ndarray = np.sqrt(x_velocities ** 2 + y_velocities ** 2 + z_velocities ** 2)
    acceleration_magnitudes: np.ndarray = np.gradient(velocity_magnitudes, time_values)

    image_path: str = './21600x21600_earth.jpg'
    figure = create_textured_sphere(image_path)

    mlab.plot3d(
        x_positions,
        y_positions,
        z_positions,
        tube_radius = 0.001 * RADIUS_OF_EARTH,
        color = (0, 1, 1),
        tube_sides = 32
    )

    mlab.show()

    # Plot altitude, velocity, and acceleration vs. time using Matplotlib
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    axs[0].plot(time_values, altitudes_msl, label="Altitude (MSL)", color='blue')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Altitude (m)')
    axs[0].set_title('Altitude vs Time')
    axs[0].grid(True)

    axs[1].plot(time_values, velocity_magnitudes, label="Velocity", color='green')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Speed (m/s)')
    axs[1].set_title('Speed vs Time')
    axs[1].grid(True)

    axs[2].plot(time_values, vertical_velocities, label="Vertical Velocity", color='red')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Vertical Velocity (m/s²)')
    axs[2].set_title('Vertical Velocity vs Time')
    axs[2].grid(True)

    axs[3].plot(time_values, acceleration_magnitudes, label="Acceleration", color='red')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Acceleration (m/s²)')
    axs[3].set_title('Acceleration vs Time')
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
