import amulet
import amulet.utils.world_utils
from amulet.api.chunk import Chunk
from amulet.api.block import Block
from amulet.level.formats import anvil_world
import math
import random
import logging
import numpy as np


level = amulet.load_level("parkour_level")
dimension="minecraft:overworld"
version=("java",(1, 20, 4))
anvil = amulet.load_format("parkour_level")
anvilversion = anvil.max_world_version
print(f"anvil version: {anvil.version}, world anvil version: {anvilversion}")

#abstractions
def block_to_chunk(x,y):
    coords=amulet.utils.world_utils.block_coords_to_chunk_coords(x,y)
    return coords

def chunk_to_block(cx,cy):
    coords=amulet.utils.world_utils.chunk_coords_to_block_coords(cx,cy)
    return coords

#filtering

lowest_world_block=-64
highest_world_block=319

def get_highest_block(x, z):
    for y in range(highest_world_block, lowest_world_block - 1, -1):
        universal_block = level.get_block(x, y, z, dimension)
        if universal_block.base_name != "air":
            return universal_block, y
    return None

def higher_than_highest(xp,yp,zp):
#checks if the proposed block is higher than the terrain at those x and z coordinates.
    if yp > get_highest_block(xp,zp)[1]:
        return True
    else:
        return False


def project_to_2d(point_3d):
    """ Project a 3D point onto the XZ plane by dropping the y-coordinate. """
    return np.array([point_3d[0], point_3d[2]])

def is_within_angle(point_a, point_b, point_c, max_angle_degrees):
    """
    Check if point_c is within a certain angle of the vector from point_a to point_b.
    Points are in 3D, but the check is done in 2D by dropping the z-coordinate.
    """
    print("checking angle")
    # Convert points to 2D
    vector_2d = np.array([point_b[0] - point_a[0], point_b[2] - point_a[2]])
    point_c_2d = np.array([point_c[0] - point_a[0], point_c[2] - point_a[2]])

    # Calculate the angle between the vector and the line from point_a to point_c in 2D
    dot_product = np.dot(vector_2d, point_c_2d)
    magnitude_vector = np.linalg.norm(vector_2d)
    magnitude_vector_c = np.linalg.norm(point_c_2d)

    # Avoid division by zero
    if magnitude_vector == 0 or magnitude_vector_c == 0:
        return False

    # Clip the value to avoid floating-point precision errors
    cos_angle = np.clip(dot_product / (magnitude_vector * magnitude_vector_c), -1.0, 1.0)
    angle_degrees = np.arccos(cos_angle) * (180.0 / np.pi)  # Convert to degrees

    # Check if the angle is within the specified range
    angle_bool = angle_degrees <= max_angle_degrees
    print(f"angle of potential block is valid: {angle_bool}")

    return angle_bool


def is_at_least_4_blocks_away(jumplist, current_point):
    """
    Checks if the current point is at least 4 blocks away from all points in the jumplist except the most recent one
    in x, y, and z coordinates.

    Args:
        jumplist (list of tuples): List of jump coordinates (x, y, z).
        current_point (tuple): The current point coordinates (x, y, z).

    Returns:
        bool: True if the current point is at least 4 blocks away from all but the most recent point in the jumplist, False otherwise.
    """
    for point in jumplist[:-1]:  # Exclude the most recent jump
        if abs(point[0] - current_point[0]) < 8 or \
           abs(point[1] - current_point[1]) < 8 or \
           abs(point[2] - current_point[2]) < 8:
            return False
    return True


def interf_check(jumplist,xp,yp,zp):
    """checks if the next jump in the sequence would interfere with previous ones.

    Args:
        jumplist (list): list of coordinate tuples of blocks
        xp (float): potential x coordinate
        yp (float): potential y coordinate
        zp (float): potential z coordinate

    Returns:
        bool: Whether or not the proposed block would block previous jumps from being possible
    """
    if len(jumplist) < 2:
        # Not enough jumps to check interference
        return False

    potential_jump = (xp, yp, zp)
    

    if is_at_least_4_blocks_away(jumplist,potential_jump):
        return True
    # Check height clearance is greater than 4
    print(f"block {potential_jump} interfered, omitting")
    return False



print(f"the highest block at the specified location is: {get_highest_block(100,10)}")

def precalculate_paraboloid_slice(max_height=319, max_angle_degrees=20, x_range=4, z_range=4, lowest_world_block=-64):
    """Precalculates a slice of the paraboloid within the maximum deviation angle."""
    paraboloid_blocks_by_y = {}
    max_angle_radians = np.radians(max_angle_degrees)
    
    for x in range(-x_range, x_range + 1):
        for z in range(-z_range, z_range + 1):
            # Check if the (x, z) point is within the angle slice
            angle = np.arctan2(abs(z), abs(x))  # Angle from the positive x-axis
            if angle <= max_angle_radians:
                y = max_height - 0.075 * (x**2 + z**2)
                if lowest_world_block <= y:
                    y_int = int(y)
                    if y_int not in paraboloid_blocks_by_y:
                        paraboloid_blocks_by_y[y_int] = []
                    paraboloid_blocks_by_y[y_int].append((x, y_int, z))
    
    return paraboloid_blocks_by_y

def rotate_point_around_origin(point, angle_radians):
    """Rotates a point around the origin by a given angle in radians."""
    cos_theta, sin_theta = np.cos(angle_radians), np.sin(angle_radians)
    x, z = point
    return cos_theta * x - sin_theta * z, sin_theta * x + cos_theta * z

def translate_and_rotate_paraboloid(blocks_by_y, xc, yc, zc, heading_angle_radians, max_height=319, lowest_world_block=-64):
    """Translates and rotates the precalculated blocks to align with the current jump position."""
    translated_rotated_blocks_by_y = {}
    jump_origin_y = max_height - 1.25

    for y, blocks in blocks_by_y.items():
        new_y = y + yc - jump_origin_y
        if lowest_world_block <= new_y <= highest_world_block:
            for x, _, z in blocks:
                rotated_x, rotated_z = rotate_point_around_origin((x, z), heading_angle_radians)
                new_x = int(rotated_x + xc)  # Convert to integer
                new_z = int(rotated_z + zc)  # Convert to integer
                translated_rotated_blocks_by_y.setdefault(new_y, []).append((new_x, new_y, new_z))

    return translated_rotated_blocks_by_y



def generate_jump(jumplist, xc, yc, zc, precalculated_blocks_by_y, low_threshold, high_threshold,min_jump_distance, max_downward_jump=3, max_angle_deviation=20):
    attempts = 0
    max_attempts = 100

    # Translate the precalculated paraboloid
    translated_blocks_by_y = translate_and_rotate_paraboloid(precalculated_blocks_by_y, xc, yc, zc, max_angle_deviation)

    # Define heading_vector outside of if-else scope
    heading_vector = np.array([1, 0])  # Default heading

    # Calculate the heading vector and second_last_jump
    if len(jumplist) >= 2:
        second_last_jump = jumplist[-2]
        last_jump = jumplist[-1]
        # Update the heading vector based on the last two jumps
        heading_vector = np.array([last_jump[0] - second_last_jump[0], last_jump[2] - second_last_jump[2]])
    else:
        # Default heading and position for the first jump or when there's only one jump
        last_jump = (xc, yc, zc) if jumplist else (0, 0, 0)
        second_last_jump = (last_jump[0] - heading_vector[0], last_jump[1], last_jump[2] - heading_vector[1])

    # Introduce a small random variation in the heading
    heading_variation = np.radians(random.uniform(-max_angle_deviation, max_angle_deviation))
    cos_var, sin_var = np.cos(heading_variation), np.sin(heading_variation)
    heading_vector = np.array([cos_var * heading_vector[0] - sin_var * heading_vector[1], sin_var * heading_vector[0] + cos_var * heading_vector[1]])


    direction_vectors = []
    for i in range(max(0, len(jumplist) - 3), len(jumplist) - 1):
        direction_vectors.append((jumplist[i + 1][0] - jumplist[i][0], jumplist[i + 1][2] - jumplist[i][2]))
    avg_direction = np.mean(direction_vectors, axis=0) if direction_vectors else np.array([1, 0])

    if direction_vectors:
        avg_direction = np.mean(direction_vectors, axis=0)
        # Introduce a slight random variation to the average direction
        variation_angle = np.radians(np.random.uniform(-max_angle_deviation, max_angle_deviation))
        avg_direction = np.array([avg_direction[0] * np.cos(variation_angle) - avg_direction[1] * np.sin(variation_angle),
                                  avg_direction[0] * np.sin(variation_angle) + avg_direction[1] * np.cos(variation_angle)])
    else:
        avg_direction = np.array([np.random.choice([-1, 1]), np.random.choice([-1, 1])])  # Random initial direction

    potential_jumps = []
    for y, blocks in translated_blocks_by_y.items():
        if abs(y-yc) <= max_downward_jump:
            for block in blocks:
                direction_to_jump = np.array([block[0] - xc, block[2] - zc])
                distance = np.sqrt((block[0] - xc)**2 + (block[2] - zc)**2)
                print(distance)
                # Check if the jump is aligned with the preferred direction and is further than minimum distance away
                if np.dot(direction_to_jump, avg_direction) > 0:
                    potential_jumps.append(block)

    print(f"potential_jumps: {potential_jumps}")

    while attempts < max_attempts and potential_jumps:
        new_jump = random.choice(potential_jumps)
        new_x, new_y, new_z = new_jump

        # Check other criteria
        if not interf_check(jumplist, new_x, new_y, new_z) and higher_than_highest(new_x, new_y, new_z):
            return new_x, new_y, new_z

        attempts += 1

    return None


precalculated_blocks_by_y=precalculate_paraboloid_slice()
#params
#iterations
iterations = 100
max_downward_jump=5
low_threshold=63
high_threshold=200

start=(10,200,10)

def generate_course_coords(iterations,start, low_threshold, high_threshold, max_downward_jump):
    jumplist = [start]
    
    for _ in range(iterations):
        # Get the last element of jumplist and extract the coordinates
        xc, yc, zc = jumplist[-1]

        # Generate the next jump
        next_jump = generate_jump(jumplist, xc, yc, zc, precalculated_blocks_by_y, low_threshold, high_threshold, max_downward_jump)

        # Check if a valid jump was generated
        if next_jump is not None:
            jumplist.append(next_jump)
        else:
            # If no valid jump was generated, break the loop
            logging.warning('Failed to generate a valid next jump. Stopping the course generation.')
            break

    print(jumplist)
    return jumplist

jump_block_type = Block("minecraft","stone")

def place_blocks(jumplist, block):
    """ Places blocks at coordinates specified in the jumplist. """
    index = 0
    for jump in jumplist:
        x, y, z = map(int, jump)  # Convert coordinates to integers

        # Ensure coordinates are within the world bounds
        if lowest_world_block <= y <= highest_world_block:
            try:
                level.set_version_block(x, y, z, dimension, version, block)
                logging.info(f"Setting block for jump {index}")
            except Exception as e:
                logging.error(f"Error setting block at ({x}, {y}, {z}): {e}")
        else:
            logging.error(f"Coordinate ({x}, {y}, {z}) is out of world bounds.")

        index += 1

# Example call to generate_course_coords

jumplist=generate_course_coords(iterations, start, low_threshold, high_threshold, max_downward_jump)
place_blocks(jumplist,jump_block_type)
level.save()
