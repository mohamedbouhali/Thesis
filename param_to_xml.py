from xml_to_param import xml_to_params_and_structure
import os
from lxml import etree as ET


def param_to_xml(xml_string, object_info,folder_path,file_name):
    """
    Create XML file based on the xml string, and then puts the given bodies in the Worldbpdy section.
    Parameters
    ----------
    xml_string: the XML structure
    folder_path:Where to write the XML file
    file_name: The name of the file


    Returns: Return the absolute path of the xml file
    -------
    """
    # first we crete the root elment of the xml file
    root = ET.fromstring(xml_string)
    #then we will change the worldbody tag by the given object infos
    worldbody = root.find('.//worldbody')

    # hier I insert the bodies inthe <worldbody> section.
    for name, body_info in object_info.items():
        body_elem = ET.SubElement(worldbody, 'body', attrib={'name': name, 'pos': body_info.pos})
        #now we use the body_elem as root and not the worldbody, because geom and joint and site are existing as sub attrbutes under the <body ...>
        ET.SubElement(body_elem, 'geom', attrib=body_info.geom['geom'])
        test=body_info.joint['joint']
        test2= body_info.site['site']
        #only if the joint and site are not None, we save them in XML (for object). For obstacle, they are empty so no need to save them
        if test:
            ET.SubElement(body_elem, 'joint', attrib=test)
        if test:
            ET.SubElement(body_elem, 'site', attrib=test2)

    #the tags must be corrctd and closed coreclty
    # we find the absolute path where to save the xml file
    file_path = os.path.join(folder_path, file_name)
    tree = ET.ElementTree(root)
    #we convert the tree to a string to save in the file
    xml_output = ET.tostring(tree, pretty_print=True, encoding='unicode')
    # I added the xml declaration because it's deleted from the function path_to_xml.
    xml_output='<?xml version="1.0" encoding="UTF-8"?>\n' + xml_output
    # we save the file in the given path
    with open(file_path, 'w', encoding='UTF-8') as file:
        file.write( xml_output)

