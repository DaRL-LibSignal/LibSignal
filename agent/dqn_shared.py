from typing import Any, Sequence
import numpy as np
from common.registry import Registry
import torch
from pfrl.agents import DQN
from pfrl.utils.contexts import evaluating

@Registry.register_model('dqn_shared')
class SharedDQN(DQN):
    def __init__(self, q_function, optimizer,replay_buffer, gamma, explorer, minibatch_size, replay_start_size, phi, target_update_interval, update_interval):

        super().__init__(q_function, optimizer, replay_buffer, gamma, explorer,
                         minibatch_size=minibatch_size, replay_start_size=replay_start_size, phi=phi,
                         target_update_interval=target_update_interval, update_interval=update_interval)

    
    
    def act(self, obs, valid_acts=None, reverse_valid=None):
        return self.batch_act(obs, valid_acts=valid_acts, reverse_valid=reverse_valid)

    def observe(self, obs: Sequence[Any], reward: Sequence[float], done: Sequence[bool], reset: Sequence[bool]) -> None:
        self.batch_observe(obs, reward, done, reset)

    def batch_act(self, batch_obs: Sequence[Any], valid_acts=None, reverse_valid=None, test=False) -> Sequence[Any]:
        if valid_acts is None: 
            return super(SharedDQN, self).batch_act(batch_obs)
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            # TODO do not use gpu
            batch_qvals = batch_av.params[0].detach().cpu().numpy() # shape: [sub_agents, 1+ob_length(remove right_lane?)]
            batch_argmax = []
            for i in range(len(batch_obs)):
                batch_item = batch_qvals[i]
                max_val, max_idx = None, None
                for idx in valid_acts[i]:
                    batch_item_qval = batch_item[idx]
                    if max_val is None:
                        max_val = batch_item_qval
                        max_idx = idx
                    elif batch_item_qval > max_val:
                        max_val = batch_item_qval
                        max_idx = idx
                batch_argmax.append(max_idx)
            batch_argmax = np.asarray(batch_argmax)

        if not test: # training
            batch_action = []
            for i in range(len(batch_obs)):
                av = batch_av[i : i + 1]
                greed = batch_argmax[i]
                act, greedy = self.explorer.select_action(self.t, lambda: greed, action_value=av, num_acts=len(valid_acts[i]))
                if not greedy:
                    act = reverse_valid[i][act]
                batch_action.append(act)

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else: # test
            batch_action = batch_argmax

        valid_batch_action = []
        for i in range(len(batch_action)):
            valid_batch_action.append(valid_acts[i][batch_action[i]])
        return valid_batch_action