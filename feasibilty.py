import trimesh
import numpy as np

def goal_in_obstacle(goal, object_center, object_size):
    """ Chekck whether the goal is inside the given object """
    #translate the created box from the origin of the coordinates system to the given center which is the correct pos
    box = trimesh.creation.box(extents=object_size*2, transform=trimesh.transformations.translation_matrix(object_center))
    contact=box.contains([goal]) # the contains fun see if the point is inside the created box
    return contact[0] #we only give one point, so we extract the first elem in the arry


def coll(center1,size1, center2, size2):
    """ Detects collision between two obstacles giver their pos and their sizes
    center1 is the position of the centre of the obstacle or the object
    size1 is the half of size of the obstacle or object (in mujoco half sizes are sored annd not complete sizes)
    Return: If there is a collision, the functions returns true
    """

    obstacle1= trimesh.creation.box(extents=size1*2)# we should by 2, because in mujoco half sizes are saved and not full size
    box2 = trimesh.creation.box(extents=size2*2)#same here
    #the boxes should moved to the cuorrct positions
    d1=center1 - size1
    d2=center2 - size2
    obstacle1.apply_translation(d1)
    box2.apply_translation(d2)
    #detetc collision
    detector = trimesh.collision.CollisionManager()
    detector.add_object("box1", obstacle1)
    detector.add_object("box2", box2)
    #return detector.in_collision_single()
    return detector.in_collision_internal()


def obstacle_not_in_table(tab_center, tab_size, center2, size2):
    """Checking if the given table conatins the given object. Only x and y coordinates are considered
    tab_center: pos of the center of the table
    tab_size: size of the table
    center2: pos of center of the obstacle
    size2: size of the obstacle
    """
    tab_size[2]=1000 # make the size in Z very big, so it always contains the obsracle
    size2[2]=0.0001 # make the size in z axix very small so that it's always conatained by the table
    table = trimesh.creation.box(extents=tab_size*2, transform=trimesh.transformations.translation_matrix(tab_center))
    obstacles = trimesh.creation.box(extents=size2*2, transform=trimesh.transformations.translation_matrix(center2))
    vertices2 = obstacles.vertices

    #for v in obstacles.vertices:
        #if not table.contains(v):
            #return True
    #return not any(table.contains(vertices2)) #one vertice is out
    return not all(table.contains(vertices2)) #if any vertice of the obstacle is not in the table, that means the obstacle is outside the table



























center1 = np.array([1.3, 0.75 ,0.2 ])
size1   = np.array([0.25 ,0.35 ,0.2])
center2 = np.array([1.3, 0.729, 0.4])
size2   = np.array([0.04, 0.025 ,0.025])
print(obstacle_not_in_table(center1, size1, center2, size2))










point = np.array([1.32, 0.89, 0.424])
center1 = np.array([1.3, 0.89, 0.42])
size1 = np.array([0.03, 0.025, 0.025])

#center2 = np.array([0.5, 1.2, 1.7])
#size2 = np.array([0.5, 0.5, 0.4])


#print(goal_in_obstacle(point, center1, size1))
#print(coll(center1,size1, center2, size2))
