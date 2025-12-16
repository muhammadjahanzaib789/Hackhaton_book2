---
sidebar_position: 1
title: "Glossary"
description: "Key terms and definitions for Physical AI robotics"
---

# Glossary

A comprehensive glossary of terms used throughout the Physical AI Book.

## A

### Action (ROS 2)
A long-running, preemptable communication pattern in ROS 2 that provides feedback during execution. Used for tasks like navigation and manipulation that take significant time.

### AMCL (Adaptive Monte Carlo Localization)
A probabilistic localization algorithm that uses particle filters to track a robot's pose in a known map. Part of Nav2.

### Arm (Robotic)
A mechanical manipulator consisting of links connected by joints, typically with an end effector for grasping or tool use.

## B

### Behavior Tree
A hierarchical control structure for robotics that uses a tree of nodes (actions, conditions, composites) to define complex behaviors. Used in Nav2 for navigation logic.

### Bridge (ROS-Gazebo)
Software that connects ROS 2 nodes with Gazebo simulation, enabling topic/service communication between the two systems.

## C

### Collision Detection
The process of determining if objects in a simulation or real environment are intersecting or about to intersect.

### Costmap
A 2D grid representation used in navigation where each cell contains a cost value indicating traversability. Higher costs indicate less desirable or impassable areas.

### CV_Bridge
A ROS package that converts between ROS image messages and OpenCV image formats.

## D

### DDS (Data Distribution Service)
The middleware standard used by ROS 2 for communication. Provides real-time, reliable, and scalable data distribution.

### Degrees of Freedom (DOF)
The number of independent parameters that define a robot's configuration. A 6-DOF arm can position and orient its end effector in 3D space.

### Depth Camera
A sensor that captures both color (RGB) and depth information for each pixel, enabling 3D scene understanding.

## E

### End Effector
The device at the end of a robotic arm, such as a gripper, tool, or sensor, that interacts with the environment.

### Encoder
A sensor that measures the position or velocity of a joint or wheel, typically using optical or magnetic principles.

## F

### FK (Forward Kinematics)
The mathematical computation of end effector position and orientation given joint angles. The "forward" direction in kinematics.

### Frame (Coordinate)
A coordinate system used to describe positions and orientations. ROS uses TF2 for frame transformations.

## G

### Gazebo
An open-source robot simulation platform that provides physics, sensors, and visualization for testing robot algorithms.

### Global Planner
The component in Nav2 that computes high-level paths from start to goal across the entire map.

### Gripper
An end effector designed to grasp and hold objects. Can be parallel jaw, vacuum, or multi-fingered.

## H

### Humble
A ROS 2 distribution (release) with long-term support. The reference distribution for this book.

### Homogeneous Transform
A 4x4 matrix that combines rotation and translation to represent pose transformations in 3D space.

## I

### IK (Inverse Kinematics)
The mathematical computation of joint angles required to achieve a desired end effector pose. More complex than FK due to potential multiple solutions.

### IMU (Inertial Measurement Unit)
A sensor combining accelerometers and gyroscopes (and sometimes magnetometers) to measure acceleration and angular velocity.

### Inference
The process of running a trained machine learning model on new data to generate predictions.

## J

### Joint
The connection between two links in a robot that allows relative motion. Can be revolute (rotational) or prismatic (linear).

### JERK (Motion)
The rate of change of acceleration. Important for smooth robot motion planning to avoid mechanical stress.

## K

### Kinematics
The study of motion without considering forces. Includes forward and inverse kinematics for robots.

### Kinova
A manufacturer of collaborative robot arms commonly used in research and education.

## L

### Latency
The time delay between an input (e.g., sensor reading) and corresponding output (e.g., motor command). Critical for real-time control.

### LiDAR
Light Detection and Ranging sensor that measures distances using laser pulses. Essential for SLAM and obstacle detection.

### Link
A rigid body in a robot connected to other links via joints. Described in URDF with geometry and inertia.

### LLM (Large Language Model)
A neural network trained on text data that can understand and generate human language. Used for natural language interfaces in robotics.

### Local Planner
The component in Nav2 that generates velocity commands to follow the global path while avoiding dynamic obstacles.

### Localization
The process of determining a robot's position and orientation within a known map.

## M

### Manipulation
The use of robotic arms and end effectors to interact with objects in the environment.

### Map
A representation of the environment used for navigation. Can be 2D (occupancy grid) or 3D (voxel/mesh).

### Message (ROS 2)
A typed data structure used for communication between ROS 2 nodes. Defined in .msg files.

### MoveIt2
The ROS 2 framework for motion planning, manipulation, and perception. Provides inverse kinematics, planning, and collision avoidance.

## N

### Nav2
The ROS 2 navigation stack providing autonomous navigation capabilities including SLAM, localization, planning, and control.

### Node (ROS 2)
A single process in ROS 2 that performs computation and communicates with other nodes via topics, services, and actions.

### Neural Network
A machine learning model inspired by biological neural networks, consisting of layers of interconnected nodes.

## O

### Occupancy Grid
A 2D map representation where each cell indicates whether that location is free, occupied, or unknown.

### Odometry
Estimation of robot position based on wheel encoders or other motion sensors. Subject to drift over time.

### Ollama
An open-source tool for running LLMs locally on a computer without cloud dependency.

## P

### Path Planning
The computation of a collision-free trajectory from a start configuration to a goal configuration.

### Perception
The robot's ability to sense and interpret its environment using cameras, LiDARs, and other sensors.

### Plugin (Gazebo)
A software module that extends Gazebo's functionality, such as adding sensors or custom physics.

### Pose
The combination of position (x, y, z) and orientation (roll, pitch, yaw or quaternion) describing an object's location.

### Publisher
A ROS 2 entity that sends messages on a topic. Multiple publishers can publish to the same topic.

## Q

### QoS (Quality of Service)
Settings in ROS 2 that define communication reliability, durability, and other properties for topics and services.

### Quaternion
A mathematical representation of 3D rotation using four numbers (x, y, z, w). Avoids gimbal lock and enables smooth interpolation.

## R

### RealSense
Intel's line of RGB-D cameras commonly used in robotics for depth perception.

### Recovery Behavior
Actions taken when navigation fails, such as backing up, clearing costmaps, or rotating in place.

### ROS 2 (Robot Operating System 2)
An open-source middleware for robotics providing tools, libraries, and conventions for building robot applications.

### RVIZ2
The 3D visualization tool for ROS 2, displaying sensor data, robot models, paths, and other information.

## S

### SDF (Simulation Description Format)
An XML format for describing worlds, models, and physics properties in Gazebo simulations.

### Service (ROS 2)
A request-response communication pattern in ROS 2 for synchronous operations that complete quickly.

### SLAM (Simultaneous Localization and Mapping)
Algorithms that build a map of an unknown environment while simultaneously tracking the robot's location within it.

### Subscriber
A ROS 2 entity that receives messages from a topic. A topic can have multiple subscribers.

## T

### TF2 (Transform Library)
The ROS 2 library for tracking coordinate frame transforms over time. Essential for relating sensor data to robot frames.

### Tokenizer
A component that converts text into numerical tokens for processing by language models.

### Topic (ROS 2)
A named bus over which nodes publish and subscribe to messages. Implements publish-subscribe pattern.

### Trajectory
A time-parameterized path specifying positions (and optionally velocities and accelerations) at each timestep.

### TurtleBot
A popular low-cost mobile robot platform used for ROS education and research.

## U

### URDF (Unified Robot Description Format)
An XML format for describing robot kinematics, visual appearance, collision geometry, and physical properties.

## V

### VLA (Vision-Language-Action)
Neural network models that take visual input and language commands to produce robot action outputs directly.

### VSLAM (Visual SLAM)
SLAM using cameras (rather than LiDAR) as the primary sensor. Includes monocular, stereo, and RGB-D variants.

### Velocity Command (cmd_vel)
The standard ROS 2 message type for commanding robot velocity, containing linear and angular components.

## W

### Waypoint
A specific location the robot should navigate through or to. Multiple waypoints form a route.

### Workspace
The volume of space a robot arm can reach. Can be defined as the set of all possible end effector positions.

### World File
A file describing a simulation environment in Gazebo, including terrain, objects, and physics properties.

## X

### Xacro
An XML macro language that extends URDF for cleaner, more modular robot descriptions.

## Y

### YAML
A human-readable data format used for ROS 2 configuration files and launch parameters.

### Yaw
Rotation about the vertical (Z) axis. One of the three Euler angles (roll, pitch, yaw).

## Z

### Zero Position
The reference configuration of a robot where all joints are at their zero angle, typically the "home" position.
