import bpy, bmesh, random, mathutils
from math import pi, tan
from mathutils import Vector
import numpy as np
from bpy import context
import math
import os
import sys
import time
import csv
import string

shape_val = sys.argv[4] #Sphere1, Sphere2, Cylinder1, Cylinder2, Disk1, Disk2
path = "."
render_path = ''


def build_bottle(label):
    bpy.ops.mesh.primitive_circle_add(radius=1,
                                      enter_editmode=False,
                                      location=(0, 0, 0))



    bottle_shell = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness = 0.02
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].render_levels = 2
    bpy.context.object.modifiers["Subdivision"].levels = 2
    sc = bpy.context.scene
    sc.rigidbody_world.enabled = True
    collection = bpy.data.collections.new("Collection_test")
    sc.rigidbody_world.collection =collection
    sc.rigidbody_world.collection.objects.link(bottle_shell)
    bottle_shell.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.rigid_body.mesh_source = 'FINAL'


    bpy.context.object.rigid_body.mass = 1
    body_length =1.9
    body_taper = .6

    neck_length = .1
    neck_taper_out = body_taper +1

    top_length = .2

    print('body length:', body_length)
    print('body taper:', body_taper)
    print('neck length:', neck_length)
    print('neck taper out:', neck_taper_out)
    print('top length:', top_length)

    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_ssr_refraction = True

    # Body length
    bpy.ops.object.mode_set(mode='EDIT')
    bmesh.from_edit_mesh(bpy.context.object.data)
    extrude(0, 0, body_length)

    cleanup_bottom()


    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    mesh = bmesh.from_edit_mesh(bpy.context.object.data)
    for v in mesh.verts:
        if v.co[2] > body_length * .99:
            v.select = True
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.editmode_toggle()
    # continue extruding

    # body taper to neck
    extrude(0, 0, 0)
    transform_resize(body_taper, body_taper, body_taper)

    # extrude neck
    extrude(0, 0, 0)
    transform_translate(0, 0, neck_length)

    # lip for bottle top
    extrude(0, 0, 0)
    transform_resize(neck_taper_out, neck_taper_out, neck_taper_out)

    # length of top piece
    extrude(0, 0, top_length)
    extrude(0, 0, 0)

    # shrink down, make last extrusion, seal shut
    transform_resize(0.586486, 0.586486, 0.586486)
    extrude(0, 0, 0)
    bpy.ops.object.mode_set(mode='OBJECT')

    set_up_glass_shader(label)

    bpy.ops.object.select_all(action='DESELECT')


def set_up_glass_shader(label):
    # Get material
    glass_mat = bpy.data.materials.get("glass")
    if glass_mat is None:
        # create material
        glass_mat = bpy.data.materials.new(name="glass")
    glass_mat.use_nodes = True
    ob = bpy.context.object
    bpy.ops.object.shade_smooth()
    if ob.data.materials:
        ob.data.materials[0] = glass_mat
    else:
        ob.data.materials.append(glass_mat)

    nodes = glass_mat.node_tree.nodes
    node = nodes.get('Glass BSDF')
    if node:
        nodes.remove(node)
    if not bpy.data.materials['glass'].node_tree.nodes.get('Glass BSDF'):
        bpy.data.materials['glass'].node_tree.nodes.new('ShaderNodeBsdfGlass')
        glass_bsdf_output = bpy.data.materials['glass'].node_tree.nodes["Glass BSDF"].outputs['BSDF']
        mat_output_surface_output = bpy.data.materials['glass'].node_tree.nodes["Material Output"].inputs['Surface']
        bpy.data.materials['glass'].node_tree.links.new(glass_bsdf_output, mat_output_surface_output)
    bpy.context.object.active_material.use_backface_culling = True
    bpy.context.object.active_material.use_screen_refraction = True
    bpy.context.object.active_material.use_sss_translucency = True
    bpy.data.materials["glass"].node_tree.nodes["Glass BSDF"].inputs[0].default_value = (
        1, 0.989, 0.957, 1)
    bpy.data.materials["glass"].node_tree.nodes["Glass BSDF"].inputs[1].default_value = 0.00
    bpy.data.materials["glass"].node_tree.nodes["Glass BSDF"].inputs[2].default_value = 1.45


    if label:
        print('wlabel')
        texImage = bpy.data.materials['glass'].node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(path + "biglabel.png")
        bpy.data.materials['glass'].node_tree.links.new(glass_bsdf_output, texImage.outputs['Color'])


    # Assign it to object
        if ob.data.materials:
            ob.data.materials[0] = bpy.data.materials['glass']
        else:
            ob.data.materials.append(bpy.data.materials['glass'])



def cleanup_bottom():
    # look at all of the faces of the cube, find the one that is 'facing' the positive direction on the y axis
    mesh = bmesh.from_edit_mesh(bpy.context.object.data)
    bpy.ops.mesh.select_all(action='DESELECT')
    for v in mesh.verts:
        if v.co[2] == 0.0:
            v.select = True
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.editmode_toggle()

    # go back and clean up the bottom
    extrude(0, 0, 0)
    transform_resize(0.794349, 0.794349, 0.794349)
    extrude(0, 0, 0)
    bpy.ops.mesh.merge(type='CENTER')


def extrude(x, y, z):
    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"use_normal_flip": False, "mirror": False},
                                     TRANSFORM_OT_translate={"value": (x, y, z),
                                                             "orient_type": 'GLOBAL',
                                                             "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                                             "orient_matrix_type": 'GLOBAL',
                                                             "constraint_axis": (False, False, False), "mirror": False,
                                                             "use_proportional_edit": False,
                                                             "proportional_edit_falloff": 'SMOOTH',
                                                             "proportional_size": 1,
                                                             "use_proportional_connected": False,
                                                             "use_proportional_projected": False, "snap": False,
                                                             "snap_target": 'CLOSEST', "snap_point": (0, 0, 0),
                                                             "snap_align": False, "snap_normal": (0, 0, 0),
                                                             "gpencil_strokes": False, "cursor_transform": False,
                                                             "texture_space": False, "remove_on_cancel": False,
                                                             "release_confirm": False, "use_accurate": False})


def transform_resize(x, y, z):
    bpy.ops.transform.resize(value=(x, y, z), orient_type='GLOBAL',
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True,
                             use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1,
                             use_proportional_connected=False, use_proportional_projected=False)


def transform_translate(x, y, z):
    bpy.ops.transform.translate(value=(x, y, z), orient_type='GLOBAL',
                                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False,
                                proportional_edit_falloff='SMOOTH', proportional_size=1,
                                use_proportional_connected=False, use_proportional_projected=False)



# delete everything in the scene
def clear_scene():
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)


def add_funnel():
    bpy.ops.mesh.primitive_torus_add(align='WORLD', location=(0, 0, 2.3), rotation=(0, 0, 0), major_radius=.75, minor_radius=0.2, abso_major_rad=1.25, abso_minor_rad=1.20)
    bpy.ops.rigidbody.object_add()
    funnel = bpy.context.active_object
    funnel.rigid_body.type = 'PASSIVE'
    bpy.context.object.hide_render = True
    bpy.context.object.rigid_body.mass = 0.5


def create_disks(num_of_ob, depth):
    #    bpy.context.object.rigid_body.enabled = False

    # Create a material
    mat1 = bpy.data.materials.new("Red")
    mat2 = bpy.data.materials.new("Blue")
    mat3 = bpy.data.materials.new("Green")
    mat4 = bpy.data.materials.new("Yellow")
    mat_list = [mat1, mat2, mat3, mat4]
    for z in range(num_of_ob):  # create Cubes

        bpy.ops.mesh.primitive_cylinder_add(radius=.5, depth=depth, enter_editmode=False, align='WORLD',
                                            location=(0, 0, .05 + .05 * z), scale=(1, 1, 1))
        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.mass = 0.2
        bpy.context.object.rigid_body.collision_shape = 'BOX'
        bpy.context.object.rigid_body.mesh_source = 'FINAL'
        obj = bpy.context.object
        obj.color = (1, 0, 0, 1)

        bpy.ops.object.modifier_add(type='COLLISION')
        bpy.context.object.collision.absorption = 1
        bpy.context.object.collision.stickiness = 1
        bpy.context.object.collision.friction_factor = 1

        mat = random.choice(mat_list)
        # Activate its nodes
        mat.use_nodes = True

        # Get the principled BSDF (created by default)
        principled = mat.node_tree.nodes['Principled BSDF']

        # Assign the color
        if mat == mat1:
            x, y, z = 1, 0, 0
        elif mat == mat2:
            x, y, z = 0, 1, 0
        elif mat == mat3:
            x, y, z = 0, 0, 1
        else:
            x, y, z = 1, 1, 0

        principled.inputs['Base Color'].default_value = (x, y, z, 1)
        #

        # Assign the material to the object
        obj.data.materials.append(mat)


def create_cylinders(num_of_ob, radius):
    #    bpy.context.object.rigid_body.enabled = False

    # Create a material
    mat1 = bpy.data.materials.new("Red")
    mat2 = bpy.data.materials.new("Blue")
    mat3 = bpy.data.materials.new("Green")
    mat4 = bpy.data.materials.new("Yellow")
    mat_list = [mat1, mat2, mat3, mat4]
    A = 100 / num_of_ob
    R = .5

    for z in range(num_of_ob):  # create Cubes

        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=2.2, enter_editmode=False, align='WORLD',
                                            location=(math.sin(A * z) * R, math.cos(A * z) * R, 2.5), scale=(1, 1, 1))

        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.mass = .2
        bpy.context.object.rigid_body.collision_shape = 'CYLINDER'
        bpy.context.object.rigid_body.mesh_source = 'FINAL'
        obj = bpy.context.object
        obj.color = (1, 0, 0, 1)

        mat = random.choice(mat_list)
        # Activate its nodes
        mat.use_nodes = True

        # Get the principled BSDF (created by default)
        principled = mat.node_tree.nodes['Principled BSDF']

        # Assign the color
        if mat == mat1:
            x, y, z = 1, 0, 0
        elif mat == mat2:
            x, y, z = 0, 1, 0
        elif mat == mat3:
            x, y, z = 0, 0, 1
        else:
            x, y, z = 1, 1, 0

        principled.inputs['Base Color'].default_value = (x, y, z, 1)
        #

        # Assign the material to the object
        obj.data.materials.append(mat)


def create_spheres(num_of_ob, radius):
#   bpy.context.object.rigid_body.enabled = False

    # Create a material
    mat1 = bpy.data.materials.new("Red")
    mat2 = bpy.data.materials.new("Blue")
    mat3 = bpy.data.materials.new("Green")
    mat4 = bpy.data.materials.new("Yellow")
    mat_list = [mat1, mat2, mat3, mat4]
    for z in range (num_of_ob): # create Cubes

        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, align='WORLD', location=(0,0,2+z*.2))
        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.mass = 0.2
        bpy.context.object.rigid_body.collision_shape = 'SPHERE'
        bpy.context.object.rigid_body.mesh_source = 'FINAL'
        obj = bpy.context.object
        obj.color = (1,0,0,1)


        mat = random.choice(mat_list)
        # Activate its nodes
        mat.use_nodes = True

        # Get the principled BSDF (created by default)
        principled = mat.node_tree.nodes['Principled BSDF']

        # Assign the color
        if mat==mat1:
            x,y,z = 1,0,0
        elif mat==mat2:
            x,y,z = 0,1,0
        elif mat==mat3:
            x,y,z = 0,0,1
        else:
            x,y,z = 1,1,0

        principled.inputs['Base Color'].default_value = (x,y,z,1)
#

        # Assign the material to the object
        obj.data.materials.append(mat)





def add_cam():

    bpy.context.scene.render.resolution_x = 1080

    # Camera 1 - (0)
    cam1 = bpy.data.cameras.new("view1")
    cam1.lens = 95
    cam_obj1 = bpy.data.objects.new("view1", cam1)
    cam_obj1.location = (0, 0 , 8)
    cam_obj1.rotation_euler = (0, 0, 1.575)


    #View 2 (90)
    cam2 = bpy.data.cameras.new("view2")
    cam2.lens = 95
    cam_obj2 = bpy.data.objects.new("view2", cam2)
    cam_obj2.location = (8.1, 0 , 1.11)
    cam_obj2.rotation_euler = (1.57325, 0, 1.57)


    #View 3 - 22.5
    cam3 = bpy.data.cameras.new("view3")
    cam3.lens = 95
    cam_obj3 = bpy.data.objects.new("view3", cam3)
    cam_obj3.location = (3, 0 , 8.5)
    cam_obj3.rotation_euler = (.3927, 0, 1.57)


    #View 4  - 45
    cam4 = bpy.data.cameras.new("view4")
    cam4.lens = 95
    cam_obj4 = bpy.data.objects.new("view4", cam4)
    cam_obj4.location = (6, 0 , 7.25)
    cam_obj4.rotation_euler = (.7854, 0, 1.57)


    #View 5 - 62
    cam5 = bpy.data.cameras.new("view5")
    cam5.lens = 95
    cam_obj5 = bpy.data.objects.new("view5", cam5)
    cam_obj5.location = (7.45, 0 , 5)
    cam_obj5.rotation_euler = (1.08, 0, 1.57)

    bpy.context.scene.collection.objects.link(cam_obj1)
    bpy.context.scene.collection.objects.link(cam_obj2)
    bpy.context.scene.collection.objects.link(cam_obj3)
    bpy.context.scene.collection.objects.link(cam_obj4)
    bpy.context.scene.collection.objects.link(cam_obj5)

    # Create light datablock
    light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
    light_data.energy = 1000

    # Create new object, pass the light data
    light_object = bpy.data.objects.new(name="my-light", object_data=light_data)


    # Link object to collection in context
    bpy.context.collection.objects.link(light_object)
#    bpy.context.collection.objects.link(cam_obj1)

    light_data1 = bpy.data.lights.new(name="my-light-data1", type='POINT')
    light_data1.energy = 1000

    # Create new object, pass the light data
    light_object1 = bpy.data.objects.new(name="my-light1", object_data=light_data1)

    bpy.context.collection.objects.link(light_object1)


    # Change light position
    light_object1.location = (4, 4, 5)

    light_object.location = (4, -4, 5)



def cam(name_camera):
    targetobj = bpy.data.objects['Circle']

    pointyobj = bpy.data.objects[name_camera]

    ttc = pointyobj.constraints.new(type='TRACK_TO')
    ttc.target = targetobj
    ttc.track_axis = 'TRACK_Z'
    ttc.up_axis = 'UP_X'

    bpy.ops.object.select_all(action='DESELECT')
    pointyobj.select_get()
    bpy.ops.object.visual_transform_apply()

    pointyobj.constraints.remove(ttc)

def id_generator(size=9, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def render_frame(fp, ob_name, cam_name, random_string):
    bpy.context.scene.camera = bpy.context.scene.objects.get(cam_name)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.frame_end = 350
    m_f = 350
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    scene = bpy.data.scenes['Scene']
    bpy.ops.ptcache.bake_all()
    scene.timeline_markers.new('F_01', frame=m_f)
    marker_frames = [m.frame for m in scene.timeline_markers]
    print(marker_frames)
    scene.frame_set(m_f)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.preview_samples = 512
    bpy.context.scene.cycles.samples = 512
    scene.render.filepath = fp + ob_name  +  '_' + random_string  + '_' + cam_name
    bpy.ops.render.render(write_still=True) # render still
    return ob_name + '_' + random_string  + '_' + cam_name

def add_img_metadata(num_of_objects, imgname, cam_view, shape, true_num):

    data = [imgname, num_of_objects, cam_view, shape, true_num]
    with open(path + 'jarexp_img_data.csv', 'a') as fd:
        write = csv.writer(fd)
        write.writerow(data)

def add_label():
    bpy.ops.import_image.to_plane(files=[{"name":"biglabel.png"}], directory=path, align_axis='Y+', relative=False)
    bpy.ops.transform.resize(value=(7, 6, 1.2), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    bpy.context.object.location[1] = 0
    bpy.context.object.location[2] = 1
    bpy.context.object.location[0] = 0.019
    bpy.context.object.rotation_euler[2] =1.43359
    bpy.context.object.location[0] = 1.06
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide()
    bpy.ops.mesh.subdivide()
    bpy.ops.mesh.subdivide()
    bpy.ops.mesh.subdivide()
    bpy.ops.mesh.subdivide()
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.modifier_add(type='SHRINKWRAP')
    bpy.context.object.modifiers["Shrinkwrap"].target = bpy.data.objects['Circle']
    bpy.context.object.modifiers["Shrinkwrap"].offset = 0.02
    bpy.ops.object.shade_smooth()


def create_stim(shape):

    build_bottle(label=False)
    # if label is True:
    #     add_label()
    #     label_tag = 'wlabel'
    # else:
    #     label_tag = 'wolabel'
    print(shape)
    if shape == 'Sphere1':
        num_of_ob = np.random.randint(50, 1200, 1)[0]
        create_spheres(num_of_ob, radius=.08)
    if shape == 'Sphere2':
        num_of_ob = np.random.randint(50, 1000, 1)[0]
        create_spheres(num_of_ob, radius=.1)
    if shape == 'Disk1':
        num_of_ob = np.random.randint(5, 60, 1)[0]
        create_disks(num_of_ob, depth=.05)
    if shape == 'Disk2':
        num_of_ob = np.random.randint(5, 50, 1)[0]
        create_disks(num_of_ob, depth=.08)
    if shape == 'Cylinder1':
        num_of_ob = np.random.randint(50, 500, 1)[0]
        create_cylinders(num_of_ob, radius=.03)
    if shape == 'Cylinder2':
        num_of_ob = np.random.randint(10, 200, 1)[0]
        create_cylinders(num_of_ob, radius=.06)


    add_cam()
    cam('view1')
    cam('view2')
    cam('view3')
    cam('view4')
    cam('view5')
    random_string = id_generator()


    imgname = render_frame(render_path, shape, 'view1', random_string)
    # if label is True:
    #     true_num = (np.sum([i.matrix_world.translation[2]>0 for i in bpy.data.objects]) - 12)
    # else:
    true_num = (np.sum([i.matrix_world.translation[2]>0 for i in bpy.data.objects]) - 11)
    add_img_metadata(num_of_ob, imgname, 'view1', shape, true_num)

    imgname = render_frame(render_path, shape, 'view2', random_string)
    add_img_metadata(num_of_ob, imgname, 'view2', shape, true_num )

    imgname = render_frame(render_path, shape, 'view3', random_string)
    add_img_metadata(num_of_ob, imgname, 'view3', shape, true_num)

    imgname = render_frame(render_path, shape, 'view4', random_string)
    add_img_metadata(num_of_ob, imgname, 'view4', shape, true_num)

    imgname = render_frame(render_path, shape, 'view5', random_string)
    add_img_metadata(num_of_ob, imgname, 'view5', shape, true_num)

if __name__ == "__main__":
    clear_scene()
    bpy.ops.rigidbody.world_add()
    studio_path = path + "cookiejar_cycles.blend"
    with bpy.data.libraries.load(studio_path) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if name.startswith("Plane")]
    # link them to scene
    scene = bpy.context.scene
    for obj in data_to.objects:
        if obj is not None:
            scene.collection.objects.link(obj)
    create_stim(shape_val)
