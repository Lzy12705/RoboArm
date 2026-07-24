import numpy as np
from robosuite.environments.manipulation.door import Door


class CurriculumDoor(Door):
    """
    Door task with an added shaping term for the door's hinge angle.

    Robosuite's default shaped reward only rewards reaching the handle and
    rotating the handle itself (the latch) -- it has no gradient toward
    actually swinging the door open, which is what _check_success() measures
    (hinge_qpos > 0.3). That gap lets a policy maximize reward by just
    reaching + holding the handle turned, without ever pushing the door
    open. This adds a matching shaped term for hinge_qpos so there's a
    gradient leading toward the actual success condition.
    """

    def reward(self, action=None):
        reward = 0.0

        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            dist = np.linalg.norm(self._gripper_to_handle)
            reward += 0.25 * (1 - np.tanh(10.0 * dist))

            if self.use_latch:
                handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)

            hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
            reward += np.clip(0.25 * (hinge_qpos / 0.3), 0.0, 0.25)

        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward
