# NOTE: This is a Blender script and should be run with Blender
# Import the required libraries
import os
import bpy
import math

bpy.context.scene.cycles.samples = 100  # Set render samples
bpy.context.scene.cycles.use_denoising = True   # Enable denoising
# Set light bounces
bpy.context.scene.cycles.max_bounces = 4
bpy.context.scene.cycles.diffuse_bounces = 2
bpy.context.scene.cycles.glossy_bounces = 2
bpy.context.scene.cycles.transmission_bounces = 2
bpy.context.scene.cycles.volume_bounces = 2
# Enable GPU rendering
bpy.context.scene.cycles.device = 'GPU'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    device.use = True
# Simplify the scene
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_add(type='DECIMATE')
        bpy.context.object.modifiers["Decimate"].ratio = 0.5
        bpy.ops.object.modifier_apply(modifier="Decimate")
# Optimize light paths
bpy.context.scene.cycles.use_fast_gi = True
# bpy.context.scene.cycles.fast_gi_method = 'AO'
bpy.context.scene.cycles.ao_bounces = 1
# Delete all pre-existing scene objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
# Specify the radius of the outer sphere containing the camera, the 3D model and the lights
sr = 5
# Add six point lights in different directions
light_pos = [(sr-1,0,0), (-sr+1,0,0), (0,sr-1,0), (0,-sr+1,0), (0,0,sr-1), (0,0,-sr+1)]
for pos in light_pos:
    bpy.ops.object.light_add(type='POINT', location=pos)
    light = bpy.context.object
    light.data.energy = 1000  # Adjust energy level as needed
# Create a UV Sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=sr, location=(0, 0, 0))
sphere = bpy.context.object
# Smooth the shading of the sphere
bpy.ops.object.shade_smooth()
# Create a new material
material = bpy.data.materials.new(name="GreyMaterial")
# Enable the use of nodes (this is necessary for Principled BSDF shader)
material.use_nodes = True
# Get the material's node tree
nodes = material.node_tree.nodes
# Get the Principled BSDF shader node
bsdf = nodes.get("Principled BSDF")
# Set the base color to white
if bsdf: bsdf.inputs['Base Color'].default_value = (0, 0, 0, 1)  # (R, G, B, Alpha)
# Assign the material to the sphere
if sphere.data.materials: sphere.data.materials[0] = material   # If the sphere already has materials, replace the first one
else: sphere.data.materials.append(material)    # If the sphere has no materials, append the new material
# Get the paths of all models from the ShapeNetCore directory
data_dir = 'C:/Hope/generated_models'
model_paths = [os.listdir(data_dir)]
# Loop to generate 10 images for each 3D model
for path in model_paths:
    # Load the 3D model
    if not os.path.isfile(path):
        continue
    else:
        bpy.ops.wm.obj_import(filepath=path)
        # Get the reference to the object
        obj = bpy.context.selected_objects[0]
        # Add a camera to the scene
        bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.context.object
        # Specify the radius of the sphere, the circumference of which the camera rotates on
        r = 3
        x = math.sqrt(r/3)
        # Setting the image resolution
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        # Setting camera locations and rotation angles
        diagonal_angle = math.degrees(math.acos(1/math.sqrt(3)))
        camera_locations = [(x,x,x)]
        camera_rotations = [(math.radians(diagonal_angle), 0, math.radians(135))]
        for i in range(len(camera_locations)):
            camera.location = camera_locations[i]
            camera.rotation_euler = camera_rotations[i]
            # Set the camera as the active camera
            bpy.context.scene.camera = camera
            # Set render output settings
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            directory = 'C:/Hope/clip_images'
            os.makedirs(directory, exist_ok=True)
            bpy.context.scene.render.filepath = f'{directory}.png'
            # Render the scene
            bpy.ops.render.render(write_still=True)
        # Remove the object from the scene
        bpy.data.objects.remove(obj, do_unlink=True)