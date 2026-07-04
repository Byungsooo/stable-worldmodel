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

Training's history frames are also sampled `frameskip` raw env-steps apart
(see scripts/train/config/data/pusht.yaml), not every raw step — for PushT's
slow per-step block motion, appending every raw call instead produces 3
near-duplicate frames the model never saw in training. `frameskip` gates
the buffer to only advance once every `frameskip` raw calls, matching the
spacing the temporal position embeddings were actually trained on.

This also tracks the real raw actions taken between sampled history frames
(via the env wrapper's `info['action']`, the action that was just executed)
and frameskip-stacks them into `hist_action`, matching exactly how the
training dataset stacks raw actions into per-position blocks (see
`stable_worldmodel/data/dataset.py`). Without this, CJEPAWorldModel.rollout()
would fall back to zeroed history-action embeddings — a real train/inference
distribution mismatch, since training always sees real, non-zero action
embeddings for history positions.
"""

from collections import deque

import numpy as np

from stable_worldmodel.policy import WorldModelPolicy


class CJEPAHistoryPolicy(WorldModelPolicy):
    def __init__(self, *args, frameskip: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_len = self.solver.model.history_len
        self.frameskip = frameskip
        self._pixel_history = None
        self._action_history = None
        self._raw_action_buffer = None
        self._step_count = None
        self._action_dim = None

    def set_env(self, env):
        super().set_env(env)
        n_envs = getattr(env, 'num_envs', 1)
        self._pixel_history = [
            deque(maxlen=self.history_len) for _ in range(n_envs)
        ]
        self._action_history = [
            deque(maxlen=max(self.history_len - 1, 0)) for _ in range(n_envs)
        ]
        self._raw_action_buffer = [[] for _ in range(n_envs)]
        self._step_count = [0 for _ in range(n_envs)]

    def _normalize_action(self, raw_action_i):
        # Mirrors the training pipeline's order of operations (normalize the
        # raw per-step action, THEN frameskip-stack) — see
        # stable_worldmodel/data/formats/lance.py's _load_slice, which
        # applies the dataset transform (incl. the action z-score scaler)
        # before Dataset.__getitem__ reshapes into frameskip blocks.
        #
        # info_dict['action'] can carry an extra leading dim per env (same
        # pattern BasePolicy._prepare_info already handles generically for
        # several keys) — reduce to the single most-recent action vector
        # before normalizing, rather than assuming a flat (act_dim,) shape.
        raw_action_i = np.asarray(raw_action_i)
        if raw_action_i.ndim > 1:
            raw_action_i = raw_action_i.reshape(-1, raw_action_i.shape[-1])[-1]
        process = getattr(self, 'process', None) or {}
        if 'action' in process:
            return process['action'].transform(raw_action_i[None, :])[0]
        return raw_action_i

    def get_action(self, info_dict, **kwargs):
        needs_flush = info_dict.get('_needs_flush')
        pixels = info_dict['pixels']
        frame = pixels[:, 0] if pixels.ndim == 5 else pixels  # (n_envs, H, W, C)
        raw_action = info_dict.get('action')  # (n_envs, act_dim) or None
        n_envs = frame.shape[0]

        if raw_action is not None and self._action_dim is None:
            self._action_dim = raw_action.shape[-1]
        act_dim = self._action_dim or 1
        n_early = max(self.history_len - 1, 0)

        for i in range(n_envs):
            if needs_flush is not None and needs_flush[i]:
                self._pixel_history[i].clear()
                self._action_history[i].clear()
                self._raw_action_buffer[i] = []
                self._step_count[i] = 0

            if len(self._pixel_history[i]) == 0:
                # Cold start: no real action history yet. Fill with zero
                # placeholders, same spirit as repeating the first observed
                # frame for pixel history — a brief, unavoidable approximation
                # at the very start of an episode.
                for _ in range(self.history_len):
                    self._pixel_history[i].append(frame[i])
                for _ in range(n_early):
                    self._action_history[i].append(
                        np.zeros(act_dim * self.frameskip, dtype=np.float32)
                    )
                self._raw_action_buffer[i] = []
            else:
                if raw_action is not None and not np.isnan(raw_action[i]).any():
                    self._raw_action_buffer[i].append(
                        self._normalize_action(raw_action[i])
                    )
                if self._step_count[i] % self.frameskip == 0:
                    if len(self._raw_action_buffer[i]) == self.frameskip:
                        block = np.concatenate(self._raw_action_buffer[i], axis=-1)
                    else:
                        block = np.zeros(act_dim * self.frameskip, dtype=np.float32)
                    if n_early > 0:
                        self._action_history[i].append(block)
                    self._raw_action_buffer[i] = []
                    self._pixel_history[i].append(frame[i])
            self._step_count[i] += 1

        stacked = np.stack(
            [np.stack(list(d), axis=0) for d in self._pixel_history], axis=0
        )  # (n_envs, history_len, H, W, C)

        new_info = dict(info_dict)
        new_info['pixels'] = stacked
        if n_early > 0:
            new_info['hist_action'] = np.stack(
                [np.stack(list(d), axis=0) for d in self._action_history], axis=0
            ).astype(np.float32)  # (n_envs, history_len-1, frameskip*act_dim)
        return super().get_action(new_info, **kwargs)
