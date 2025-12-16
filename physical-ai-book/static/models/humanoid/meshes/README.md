# Humanoid Robot Mesh Files

This directory contains visual and collision mesh files for the 21-DOF humanoid robot.

## File Structure

```
meshes/
├── visual/           # High-detail meshes for rendering
│   ├── torso.dae
│   ├── head.dae
│   ├── upper_arm.dae
│   ├── forearm.dae
│   ├── hand.dae
│   ├── thigh.dae
│   ├── shin.dae
│   └── foot.dae
│
└── collision/        # Simplified meshes for physics
    ├── torso.stl
    ├── head.stl
    ├── upper_arm.stl
    ├── forearm.stl
    ├── hand.stl
    ├── thigh.stl
    ├── shin.stl
    └── foot.stl
```

## Mesh Specifications

### Visual Meshes (COLLADA .dae)
- Format: COLLADA 1.4.1
- Units: Meters
- Up-axis: Z
- Polygon count: 1000-5000 per part
- Textures: UV-mapped, 512x512 PNG

### Collision Meshes (STL)
- Format: Binary STL
- Units: Meters
- Polygon count: 50-200 per part (simplified)
- Convex hulls preferred for stability

## Using with URDF

Reference meshes in URDF:
```xml
<visual>
  <geometry>
    <mesh filename="package://physical_ai_book/static/models/humanoid/meshes/visual/torso.dae"/>
  </geometry>
</visual>
<collision>
  <geometry>
    <mesh filename="package://physical_ai_book/static/models/humanoid/meshes/collision/torso.stl"/>
  </geometry>
</collision>
```

## Generating Placeholder Meshes

For development without custom meshes, use primitive shapes in URDF (cylinders, boxes).
The provided URDF uses primitives and will work without these mesh files.

To create production meshes:
1. Use Blender or similar 3D software
2. Export visual meshes as COLLADA (.dae)
3. Create simplified collision hulls
4. Export collision meshes as STL

## Notes

- The URDF file uses geometric primitives by default
- Mesh files are optional for basic functionality
- Add meshes for improved visual realism
- Collision meshes significantly impact physics performance
