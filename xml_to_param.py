from lxml import etree as ET
import copy
import re

class BodyInfo:
    def __init__(self, pos, geom, joint, site):
        """ Data structure to save the position, the geometry attributes, the joint attributes and site attributes for the bodies inside the worldbody tag"""
        self.pos = pos
        self.geom = {"geom": geom}
        self.joint = {"joint":joint}
        self.site = {"site": site}
def xml_to_params_and_structure(path_to_xml,env):
    try:
        parser = ET.XMLParser(remove_blank_text=True)
        t = ET.parse(path_to_xml, parser)
        original_root = t.getroot()
    except Exception as x:
        raise Exception(f"Error while parsing: {x}")

    # we keep a seperate copy of the root, so we don't modify the xml file now
    r = copy.deepcopy(original_root)
    # we need the worldbody tag, as it contains all needed bodies
    worldbody = r.find('.//worldbody')
    object_bodies = {}

    # Remove specific body elements
    if worldbody is not None:
        obj_obs = [body for body in worldbody if ( body.get('name', '').startswith('object') or body.get('name', '').startswith('obstacle'))]
        for body in obj_obs:
            pos = body.get('pos', 'not given in the xml file. Please check the XML file.')
            if body.get('name', '').startswith('object'):# we extract the position of the object from the gym env and not from the xml because the gym library modifies and overwrite the position of the object in the XML file. For other obstacles, we can extract from the xml file.
                pos=env.sim.data.get_joint_qpos('object0:joint')[:3]
                pos=' '.join(map(str, pos))#convert it to string
            # we initiliaze empty attributes for geom,site and joint. If it's obstacle and not object, the joint and site will stay empty.
            g_att = {}
            j_att = {}
            s_att = {}
            #we extract the attributes of geom and joint and site from the xml file
            geom_element = body.find('geom')
            joint_element = body.find('joint')
            site_element = body.find('site')

            #now store the extracted attributes from the xml in the BodyInfo data structure
            if geom_element is not None:
                g_att = {attr: geom_element.get(attr) for attr in geom_element.keys()}
            if (joint_element is not None and body.get('name', '').startswith('object')) :
                j_att = {attr: joint_element.get(attr) for attr in joint_element.keys()}
            if (site_element is not None and body.get('name', '').startswith('object')) :
                s_att = {attr: site_element.get(attr) for attr in site_element.keys()}
            #the name of the body will be the key, and the value is the BodyInfo data structure
            object_bodies[body.get('name')] = BodyInfo(pos, g_att, j_att, s_att)
            worldbody.remove(body)

    # we convert the xml wihtout bodies to string, because we need it later to reuild the xml file
    newStrWithBodies = ET.tostring(r, pretty_print=True, encoding='unicode', method='xml')

    return newStrWithBodies, object_bodies

