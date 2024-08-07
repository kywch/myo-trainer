import numpy as np
import torch
import torch.nn as nn

import pufferlib
from pufferlib.pytorch import layer_init


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Policy(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
            )
            self.decoder_logstd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))
        else:
            self.decoders = torch.nn.ModuleList(
                [layer_init(torch.nn.Linear(hidden_size, n)) for n in env.single_action_space.nvec]
            )

        self.value_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        """Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers)."""
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1)
        return self.encoder(observations.float()), None

    def decode_actions(self, hidden, lookup, concat=True):
        """Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers)."""
        value = self.value_head(hidden)

        if self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            # batch = hidden.shape[0]
            return probs, value
        else:
            actions = [dec(hidden) for dec in self.decoders]
            return actions, value
