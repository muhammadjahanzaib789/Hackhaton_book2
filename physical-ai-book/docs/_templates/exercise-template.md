---
sidebar_position: 99
title: "Chapter X Exercises"
description: "Hands-on exercises to reinforce chapter concepts"
---

# Chapter X Exercises

## Overview

These exercises reinforce the concepts learned in Chapter X. Complete them in order, as later exercises build on earlier ones.

**Estimated Time**: 2-3 hours
**Difficulty**: Beginner → Intermediate → Advanced

## Exercise 1: [Basic Concept]

### Objective

[What the learner will accomplish]

### Difficulty

:star: Beginner (15-20 minutes)

### Background

Brief context about why this exercise matters.

### Instructions

1. First step
2. Second step
3. Third step

### Starter Code

```python
#!/usr/bin/env python3
"""
Exercise 1: [Name]
Physical AI Book - Chapter X

TODO: Complete the implementation
"""

import rclpy
from rclpy.node import Node


class ExerciseNode(Node):
    def __init__(self):
        super().__init__('exercise_node')
        # TODO: Add your implementation here
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ExerciseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Success Criteria

- [ ] Criterion 1 (specific, measurable)
- [ ] Criterion 2
- [ ] Criterion 3

### Hints

<details>
<summary>Hint 1</summary>

First hint to help if stuck.

</details>

<details>
<summary>Hint 2</summary>

Second, more specific hint.

</details>

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
#!/usr/bin/env python3
"""
Exercise 1 Solution
"""

import rclpy
from rclpy.node import Node


class ExerciseNode(Node):
    def __init__(self):
        super().__init__('exercise_node')
        # Complete implementation
        self.get_logger().info('Exercise complete!')


def main(args=None):
    rclpy.init(args=args)
    node = ExerciseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**Explanation**: Why this solution works and key learning points.

</details>

---

## Exercise 2: [Intermediate Concept]

### Objective

[What the learner will accomplish]

### Difficulty

:star::star: Intermediate (30-45 minutes)

### Background

Brief context about why this exercise matters and how it builds on Exercise 1.

### Instructions

1. First step
2. Second step
3. Third step

### Requirements

- Must use concept from Exercise 1
- Must implement feature X
- Must handle edge case Y

### Starter Code

```python
# Starter code for Exercise 2
```

### Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3
- [ ] Criterion 4

### Hints

<details>
<summary>Hint 1</summary>

Hint content.

</details>

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
# Solution code
```

</details>

---

## Exercise 3: [Advanced Challenge]

### Objective

[What the learner will accomplish]

### Difficulty

:star::star::star: Advanced (45-60 minutes)

### Background

Context about this challenging exercise.

### Instructions

1. Step one
2. Step two
3. Step three

### Requirements

- Builds on Exercises 1 and 2
- Requires integration of multiple concepts
- Must meet performance criteria

### Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Node A  │────▶│  Node B  │────▶│  Node C  │
└──────────┘     └──────────┘     └──────────┘
```

### Starter Code

```python
# Starter code for Exercise 3
```

### Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3
- [ ] Criterion 4
- [ ] Criterion 5

### Hints

<details>
<summary>Hint 1</summary>

Architectural hint.

</details>

<details>
<summary>Hint 2</summary>

Implementation hint.

</details>

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
# Solution code
```

</details>

---

## Bonus Challenge: [Extension]

### Objective

Optional advanced challenge for learners who want to go further.

### Difficulty

:star::star::star::star: Expert (1-2 hours)

### Description

Description of the bonus challenge. This should extend concepts significantly.

### Requirements

- All previous exercises completed
- Additional research may be required
- Creative solutions encouraged

### Success Criteria

- [ ] Advanced criterion 1
- [ ] Advanced criterion 2
- [ ] Creative solution implemented

---

## Self-Assessment

After completing these exercises, you should be able to:

- [ ] Skill 1 demonstrated in Exercise 1
- [ ] Skill 2 demonstrated in Exercise 2
- [ ] Skill 3 demonstrated in Exercise 3
- [ ] Integration skill demonstrated in Bonus

## Next Steps

- Review the [Chapter X Summary](./lesson-05.md)
- Proceed to [Chapter Y](../chapter-y/lesson-01.md)
- Explore [Advanced Topics](#)
