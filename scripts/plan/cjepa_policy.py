"""WorldModelPolicy variant that maintains a multi-frame pixel history.

WorldModelPolicy.get_action passes whatever `pixels` shape it receives
straight through to the solver — a single current frame per env, since
World only ever supplies one frame per step. CJEPAWorldModel was trained
with history_len=3 (baked into its temporal position embedding table), so
feeding it a single frame would silently misalign temporal position
embeddings instead of crashing. This subclass keeps a per-env ring buffer
of the last `history_len` frames and stacks them before delegating to the
base class, which handles everything else (CEM solving, receding-horizon
action buffer, warm start) unchanged.
"""

from collections import deque

import numpy as np

from stable_worldmodel.policy import WorldModelPolicy


class CJEPAHistoryPolicy(WorldModelPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_len = self.solver.model.history_len
        self._pixel_history = None

    def set_env(self, env):
        super().set_env(env)
        n_envs = getattr(env, 'num_envs', 1)
        self._pixel_history = [
            deque(maxlen=self.history_len) for _ in range(n_envs)
        ]

    def get_action(self, info_dict, **kwargs):
        needs_flush = info_dict.get('_needs_flush')
        pixels = info_dict['pixels']
        frame = pixels[:, 0] if pixels.ndim == 5 else pixels  # (n_envs, H, W, C)
        n_envs = frame.shape[0]

        for i in range(n_envs):
            if needs_flush is not None and needs_flush[i]:
                self._pixel_history[i].clear()
            if len(self._pixel_history[i]) == 0:
                for _ in range(self.history_len):
                    self._pixel_history[i].append(frame[i])
            else:
                self._pixel_history[i].append(frame[i])

        stacked = np.stack(
            [np.stack(list(d), axis=0) for d in self._pixel_history], axis=0
        )  # (n_envs, history_len, H, W, C)

        new_info = dict(info_dict)
        new_info['pixels'] = stacked
        return super().get_action(new_info, **kwargs)
