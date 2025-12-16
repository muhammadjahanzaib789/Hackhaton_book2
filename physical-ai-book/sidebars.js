// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  bookSidebar: [
    'intro',
    {
      type: 'category',
      label: '1. ROS 2 Fundamentals',
      link: {
        type: 'generated-index',
        title: 'ROS 2 Fundamentals',
        description: 'Learn ROS 2 as the robotic nervous system for humanoid control.',
        slug: '/chapter-01-ros2-fundamentals',
      },
      items: [
        'chapter-01-ros2-fundamentals/lesson-01-introduction',
        'chapter-01-ros2-fundamentals/lesson-02-installation',
        'chapter-01-ros2-fundamentals/lesson-03-nodes-topics',
        'chapter-01-ros2-fundamentals/lesson-04-services-actions',
        'chapter-01-ros2-fundamentals/lesson-05-urdf-basics',
        'chapter-01-ros2-fundamentals/exercises',
      ],
    },
    {
      type: 'category',
      label: '2. Simulation',
      link: {
        type: 'generated-index',
        title: 'Humanoid Simulation',
        description: 'Build and test humanoid robots in physics simulation.',
        slug: '/chapter-02-simulation',
      },
      items: [
        'chapter-02-simulation/lesson-01-gazebo-intro',
        'chapter-02-simulation/lesson-02-urdf-deep-dive',
        'chapter-02-simulation/lesson-03-physics-config',
        'chapter-02-simulation/lesson-04-sensor-simulation',
        'chapter-02-simulation/lesson-05-ros2-gazebo-bridge',
        'chapter-02-simulation/exercises',
      ],
    },
    {
      type: 'category',
      label: '3. Perception & Vision',
      link: {
        type: 'generated-index',
        title: 'Perception & Vision',
        description: 'Enable robots to see and understand their environment.',
        slug: '/chapter-03-perception',
      },
      items: [
        'chapter-03-perception/lesson-01-computer-vision',
        'chapter-03-perception/lesson-02-object-detection',
        'chapter-03-perception/lesson-03-isaac-perception',
      ],
    },
    {
      type: 'category',
      label: '4. Navigation',
      link: {
        type: 'generated-index',
        title: 'Navigation & Planning',
        description: 'Autonomous navigation with SLAM and Nav2.',
        slug: '/chapter-04-navigation',
      },
      items: [
        'chapter-04-navigation/lesson-01-slam',
        'chapter-04-navigation/lesson-02-nav2',
        'chapter-04-navigation/lesson-03-path-planning',
      ],
    },
    {
      type: 'category',
      label: '5. Manipulation',
      link: {
        type: 'generated-index',
        title: 'Manipulation & Control',
        description: 'Arm control, inverse kinematics, and grasping.',
        slug: '/chapter-05-manipulation',
      },
      items: [
        'chapter-05-manipulation/lesson-01-kinematics',
        'chapter-05-manipulation/lesson-02-moveit2',
        'chapter-05-manipulation/lesson-03-grasping',
      ],
    },
    {
      type: 'category',
      label: '6. LLM Integration',
      link: {
        type: 'generated-index',
        title: 'LLM Integration',
        description: 'Connect Large Language Models to robot control.',
        slug: '/chapter-06-llm-integration',
      },
      items: [
        'chapter-06-llm-integration/lesson-01-llm-basics',
        'chapter-06-llm-integration/lesson-02-task-planning',
        'chapter-06-llm-integration/lesson-03-voice-interaction',
      ],
    },
    {
      type: 'category',
      label: '7. VLA Models',
      link: {
        type: 'generated-index',
        title: 'Vision-Language-Action',
        description: 'End-to-end embodied AI with VLA models.',
        slug: '/chapter-07-vla',
      },
      items: [
        'chapter-07-vla/lesson-01-vla-introduction',
        'chapter-07-vla/lesson-02-vla-deployment',
      ],
    },
    {
      type: 'category',
      label: '8. Capstone Project',
      link: {
        type: 'generated-index',
        title: 'Capstone Integration',
        description: 'Build an autonomous home assistant robot.',
        slug: '/chapter-08-capstone',
      },
      items: [
        'chapter-08-capstone/lesson-01-project-overview',
        'chapter-08-capstone/lesson-02-implementation',
        'chapter-08-capstone/lesson-03-integration',
        'chapter-08-capstone/lesson-04-testing',
      ],
    },
    {
      type: 'category',
      label: 'Resources',
      link: {
        type: 'generated-index',
        title: 'Resources',
        description: 'Glossary, learning paths, and quick reference materials.',
        slug: '/resources',
      },
      items: [
        'resources/glossary',
        'resources/learning-paths',
        'resources/quick-reference',
      ],
    },
  ],
};


export default sidebars;
