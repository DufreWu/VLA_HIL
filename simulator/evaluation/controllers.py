"""Reference policies use ground-truth state; they are not VLA policies."""
import numpy as np


class ScriptedPolicy:
    def reset(self, scene, task, initial, target):
        self.scene, self.task = scene, task
        self.initial, self.target = np.asarray(initial), np.asarray(target)
        self.start_hand = np.asarray(scene.robot.get_current_state()[1]).reshape(3)
        # Hand-to-fingertip offset used by the upstream Franka example.
        grasp = self.initial + [0, 0, .10]
        if task == 'grasping':
            self.waypoints = [(2., grasp + [0, 0, .15], False),
                              (4., grasp, False), (5.5, grasp, True),
                              (8.5, grasp + [0, 0, .15], True)]
        else:
            direction = self.target - self.initial
            direction[2] = 0
            direction /= np.linalg.norm(direction)
            # Closed fingertips approach behind the cube, then push across its centre.
            behind = self.initial - direction*.07 + [0, 0, .10]
            end = self.target - direction*.035 + [0, 0, .10]
            self.waypoints = [(2., behind + [0, 0, .15], True),
                              (4., behind, True), (9., end, True),
                              (10., end - direction*.06, True),
                              (12., end - direction*.06 + [0, 0, .15], True)]

    def act(self, elapsed):
        robot = self.scene.robot
        previous_t, previous_pos = 0., self.start_hand
        for end_t, end_pos, closed in self.waypoints:
            if elapsed <= end_t:
                u = np.clip((elapsed - previous_t)/(end_t - previous_t), 0, 1)
                u = u*u*(3-2*u)
                goal = previous_pos + u*(end_pos-previous_pos)
                break
            previous_t, previous_pos = end_t, end_pos
        else:
            _, goal, closed = self.waypoints[-1]
        robot.set_end_effector_pose(position=goal, orientation=robot.get_downward_orientation(),
                                   ik_method='damped-least-squares')
        if closed:
            robot.close_gripper()
        else:
            robot.open_gripper()
        return self.scene.command()
