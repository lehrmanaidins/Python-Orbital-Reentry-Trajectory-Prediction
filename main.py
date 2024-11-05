from typing import Final
import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
from scipy.integrate import solve_ivp

# Constants
GRAVITATIONAL_CONSTANT: Final[float] = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
MASS_OF_EARTH: Final[float] = 3.7327282e23  # Mass of Earth (kg)
RADIUS_OF_EARTH: Final[float] = 1_592_750  # Radius of the Earth (m)


def equations(t, y):
    x, y, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y**2 + z**2)  # Distance from the center of the Earth

    # Acceleration due to gravity
    ax = -GRAVITATIONAL_CONSTANT * MASS_OF_EARTH * x / r**3
    ay = -GRAVITATIONAL_CONSTANT * MASS_OF_EARTH * y / r**3
    az = -GRAVITATIONAL_CONSTANT * MASS_OF_EARTH * z / r**3

    return [vx, vy, vz, ax, ay, az]


# Function to create Earth with texture
def auto_sphere(image_file):
    # Create figure window
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(600, 600))

    # Load and map the texture
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)

    # Create the sphere source with texture
    R = 1  # Adjust for desired radius
    Nrad = 180
    sphere = tvtk.TexturedSphereSource(radius=R, theta_resolution=Nrad, phi_resolution=Nrad)
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    fig.scene.add_actor(sphere_actor)

    return fig


def main() -> None:
    # Initial conditions: position (x0, y0, z0) and velocity (vx0, vy0, vz0)
    initial_conditions = [RADIUS_OF_EARTH + 450_000, 0, 0, 0, 3_500, 0]

    # Time span for the integration
    time_span = (0, 5_000)
    time_eval = np.linspace(time_span[0], time_span[1], 1_000)  # Time points for evaluation

    # Solve the system
    solution = solve_ivp(equations, time_span, initial_conditions, t_eval=time_eval, method='RK23')

    # Extract positions for the trajectory
    x_positions = solution.y[0]
    y_positions = solution.y[1]
    z_positions = solution.y[2]

    # Render the Earth sphere with texture
    image_file = '21600x21600_earth.jpg'  # Replace with the path to your Earth texture image
    fig = auto_sphere(image_file)

    # Scale trajectory points to be in the same range as the Earth sphere
    x_positions /= RADIUS_OF_EARTH
    y_positions /= RADIUS_OF_EARTH
    z_positions /= RADIUS_OF_EARTH

    # Plot the rocket's trajectory
    mlab.plot3d(x_positions, y_positions, z_positions, tube_radius=0.005, color=(0, 1, 1), tube_sides=32)

    # Show the plot
    mlab.show()


if __name__ == '__main__':
    main()
