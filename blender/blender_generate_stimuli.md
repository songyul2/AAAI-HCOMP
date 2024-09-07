- Replace variables values in generate_stimuli.py file:
    1. path
    2. render_path

- Install the following addons in Blender:
    Images as Planes

- To install add-on in Blender: Go to Preferences --> Add-ons --> Search for 'Images as Planes' --> Install

- To run Blender 3.3 using terminal:
  Go to folder Blender Foudation (Usually in Program Files) -->Blender 3.3 --> 3.3

- Run: "blender.exe" -b --python generate_stimuli.py shape_val  [where shape_val can take values "Sphere1", "Sphere2", "Disk1", "Disk2", "Cylinder1", "Cylinder2"]

for example: "blender.exe" -b --python generate_stimuli.py "Disk1"