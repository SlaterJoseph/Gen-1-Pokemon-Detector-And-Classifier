import bpy
import math
import random

from mathutils import Vector


def random_lighting(light_obj: bpy.types.Object) -> None:
    """
    Randomize the location of the light source
    :param light_obj: The light source
    :return: None
    """
    elevation = random.randint(25, 70)
    azimuth = random.randint(0, 360)
    light_obj.rotation_euler = (math.radians(elevation), 0, math.radians(azimuth))

    brightness = random.uniform(4.5, 7.5)
    light_obj.data.energy = brightness



def adjust_camera(camera_obj: bpy.types.Object) -> None:
    """
    Randomize the camera angle
    :param camera_obj: The camera source
    :return: None
    """
    distance = random.uniform(4.0, 6.0)
    elevation = math.radians(random.randint(20, 55))
    azimuth = math.radians(random.randint(0, 360))

    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)

    camera_obj.location =  Vector((x, y, z))

    direction = -camera_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()

    camera_obj.data.angle = math.radians(random.uniform(35, 50))