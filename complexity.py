import numpy
import numpy as np
import trimesh
import numpy as np
from xml_to_param import  xml_to_params_and_structure

def convert_str2arr(string):
    """
    Convert a given string of values like " 0.1 0.2 0.5" to a numpy array [0.1, 0.2, 0.5]
    """
    n = string.split()
    a = np.array([float(num) for num in n])
    return a


def create_box(center, half_size):
    """
    Creates a trimesh box.
    Center: the center of the box
    half_size: half of the size of each achse
    Return: Instance of trimesh box
    """
    fs = [2 * hs for hs in half_size] #first get the full size
    transform = trimesh.transformations.translation_matrix(center)# change the position from the origin of coordinates system to the center of the object
    #folloed the steps from https://github.com/mikedh/trimesh/blob/108f7f3ff457c479bdda6c819cc51a03d6fefba9/examples/widget.py#L33
    return trimesh.creation.box(extents=fs, transform=transform)


def calculate_dis_blocked_by_obstacles(target_point,object_point,center_obstacle,half_size_obstacle):
    """
    Calculate the distance between the points of intersection of the obstacel and the ray between goal and the object that should be pushed
    If no intersection, it returns 0
    """
    # Calculate ray direction from object point to target point
    ray_direction = target_point - object_point
    x=np.linalg.norm(ray_direction)
    ray_direction = ray_direction /x   # we have to make the vector normalized toget correct result
    #followed the steps from tutorial: https://github.com/mikedh/trimesh/blob/main/examples/ray.py


    # create obstacle
    obstacle = create_box(center_obstacle, half_size_obstacle)

    # intersects_location gives the points of intersections between the obstacle and and the ray
    locations = obstacle.ray.intersects_location(
        ray_origins=[object_point],
        ray_directions=[ray_direction],
    )


    points_between_object_and_target =[] #points that are actuelly between target and object and not in the indefinite line
    for location in locations[0]:
        if np.linalg.norm(location - object_point) <= x:
            points_between_object_and_target.append(location)
    #print("points_between_object_and_target are", points_between_object_and_target.__len__())
    if points_between_object_and_target.__len__() > 1:  # points_between_object_and_target contains the intersection points4 that are between object and targets
        #print("Obstacle intersects at points", locations[0])
        # Calculate distances between each consecutive intersection point
        if points_between_object_and_target[0].shape[0] == 1: # in this case , the line only intercept the object in one point as the ray go in one direct, object-->target, which means that obstacle is actually placed in the object !
            print("The obstacle is in collision with the target or with the object to be moved, please change the pos of the object or the obstacle")
        #get the distance between these points
        distances = np.diff(points_between_object_and_target, axis=0) # we calculate the distance between the points of intersections
        distances = (np.linalg.norm(distances, axis=1)[0]) # we get the absolut value the distance

        #print("Distances between intersection points:", distances)
        return distances
    else:
        #print("No intersection with the obstacle.")
        distances=0
        return distances


def get_complexity(args, env, obstacles_info):
    """
    Calculates the complexity of the enviroment
    Parameters
    ----------
    args: args given from user
    env : the enviroment (created by make_env() )
    obstacles_info: dictionary conating the infos about the obstacles and also the object to move

    Returns : The complexity
    -------
    """
    #Extract the position of the goal and the numbe of obstacles
    goal =env.goal
    goal =goal[:3]
    num_obstacles=0
    for name, obstacle_info in obstacles_info.items():
        if name.startswith('obstacle'):
            num_obstacles+=1
    MAX_NUMBER_OF_OBSTACLES=10
    # Extract the pos of the object to push/pick/slide from the XML, if it's reach , we consider the initial pos of the grippe as, the object that should be moved
    if args.env[:7]=="FetchRe": # extract the pos from gym
        object_position=env.initial_gripper_xpos[:3]
    else :
        object_position = (env.sim.data.get_joint_qpos('object0:joint'))[:3] #we extract the position from the gym environment

    # for push and slide, the max distance between the object and the goal is the diametre of the table
    MAX_DISTANCE_BETWEEN_GOAL_IN_THE_TABLE_AND_OBJECT = numpy.sqrt(0.35 ** 2 + 0.25 ** 2)  # the size of the tanle is 0.25 0.35 0.2
    # for Reach and Pick_and_place the maximum distance is the last point that the robot can reach in the air
    # the farest point that robot can reach is approximately 0.5 0.85 1.63. With the lowest point in the corner of the table is 1.55 1.1 0.4, we calculate the length of the line connecting these 2 points
    MAX_DISTANCE_BETWEEN_GOAL_IN_THE_AIR_AND_OBJECT = numpy.sqrt((1.55 - 0.5) ** 2 + (1.1 - 0.85) ** 2 + (1.63 - 0.4) ** 2)

    # Calculate how far the goal is.
    if args.env[:7]=="FetchRe":
        farness_of_goal=np.linalg.norm(goal -  env.initial_gripper_xpos[:3])#distance between initial pos of robot and the goal, the inital position of the gripper is approximately 1.34786948 0.74894948 0.41363773
    else:
        farness_of_goal = np.linalg.norm(goal - object_position) #distance bwtween the goal and the object to move


    distance_blocked_by_obstacles=0
    for name, obstacle_info in obstacles_info.items():
        if name.startswith('obstacle'):
            distance_blocked_by_obstacles+=calculate_dis_blocked_by_obstacles(goal,object_position, convert_str2arr(obstacle_info.pos)  ,convert_str2arr(obstacle_info.geom['geom']['size']))

    surface_or_height=0
    if args.env[:7] == "FetchPu" or args.env[:7] == "FetchSl":
        for name, obstacle_info in obstacles_info.items():
            if name.startswith('obstacle'): #only obstacles are inluded and not the object to move
                x_size=(convert_str2arr(obstacle_info.geom['geom']['size'])[0])
                surface_or_height += x_size * (convert_str2arr(obstacle_info.geom['geom']['size'])[1]) # we only intereseted in the x and y sizes, not the z diemnsions
    if args.env[:7] == "FetchRe" or args.env[:7] == "FetchPi": #all dimensions are included to calculate the volume of the obstacle
        for name, obstacle_info in obstacles_info.items():
            if name.startswith('obstacle'):
                surface_or_height +=((convert_str2arr(obstacle_info.geom['geom']['size'])[0]) * (convert_str2arr(obstacle_info.geom['geom']['size'])[1])) *(convert_str2arr(obstacle_info.geom['geom']['size'])[2])
                # we are interested in all diemnsions
    complexity =(num_obstacles*0.25)/(MAX_NUMBER_OF_OBSTACLES) + (surface_or_height*0.35)/(0.25 * 0.35) + (0.4*distance_blocked_by_obstacles)/(farness_of_goal) #every metric has different weight
    return complexity






