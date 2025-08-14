# Genesis Migration Notes

## Overview
This document describes the migration from PyBullet to Genesis physics engine in `sim_genesis.py`.

## File Structure
- `sim.py` - Original PyBullet implementation (default)
- `sim_pybullet.py` - Backup of original PyBullet implementation  
- `sim_genesis.py` - New Genesis implementation

## Key Changes

### 1. Physics Engine Initialization
- **PyBullet**: Used `p.connect()` and `p.setTimeStep()`
- **Genesis**: Uses `gs.init()`, `gs.Scene()`, and `gs.world.set_time_step()`

### 2. Object Loading and Management
- **PyBullet**: Used `p.loadURDF()` returning body IDs
- **Genesis**: Uses `scene.add_entity()` with various material types (RigidBody, SoftBody, etc.)
- Objects are stored in `self.genesis_objects` dictionary instead of `self.urdf_ids`

### 3. Simulation Stepping
- **PyBullet**: `p.stepSimulation()`
- **Genesis**: `world.step()`

### 4. Robot Control
- **PyBullet**: Used joint motor control with `p.setJointMotorControlArray()`
- **Genesis**: Uses entity-based control with `set_dofs_position()` and `set_pos()`/`set_quat()`

### 5. Collision Detection
- **PyBullet**: `p.getClosestPoints()` for collision detection
- **Genesis**: Simplified collision handling using bounding box overlap checks

### 6. Camera and Rendering
- **PyBullet**: `p.getCameraImage()` with view/projection matrices
- **Genesis**: Scene-based camera system with `scene.add_camera()` and `scene.render()`

## Important Implementation Notes

### Simplified Components
Due to the complexity of the original PyBullet implementation, several components were simplified:

1. **Robot IK**: Direct pose setting instead of full inverse kinematics
2. **Contact Detection**: Simplified contact handling for suction gripper
3. **Joint Control**: Basic DOF control instead of detailed motor control
4. **Link State Tracking**: Placeholder implementation for articulated objects

### Genesis-Specific Features
The implementation takes advantage of Genesis's:
- Unified entity system for rigid and soft bodies
- Built-in material property management
- Scene-based rendering system
- Automatic collision handling

## Usage

Replace the import in your code:
```python
# Old
from manipulation.sim import SimpleEnv

# New  
from manipulation.sim_genesis import SimpleEnv
```

The API remains the same, but now uses Genesis as the backend physics engine.

## Future Improvements

1. **Full Robot IK**: Implement proper inverse kinematics using Genesis's solver
2. **Advanced Contact Detection**: Use Genesis's contact sensors for suction gripper
3. **Material Properties**: Leverage Genesis's advanced material system
4. **Soft Body Support**: Add support for deformable objects
5. **Performance Optimization**: Use Genesis's GPU acceleration features

## Dependencies

Ensure you have Genesis installed:
```bash
pip install genesis-world
```

**Important**: Genesis requires either CUDA or Vulkan backend on Linux. The CPU backend is not supported. Ensure you have:
- NVIDIA GPU with CUDA drivers (for CUDA backend), or  
- Vulkan SDK installed (for Vulkan backend)

The Genesis documentation is available at: https://genesis-world.readthedocs.io/