import numpy as np


class MicrogridStep:
    def __init__(self):
        self._obs = dict()
        self._reward = 0.0
        self._done = False
        self._info = dict(absorbed_energy=[], provided_energy=[])

    def append(self, module_name, obs, reward, done, info):
        try:
            self._obs[module_name].append(obs)
        except KeyError:
            self._obs[module_name] = [obs]
        self._reward += reward
        if done:
            self._done = True
        for key, value in info.items():
            try:
                self._info[key].append(value)
            except KeyError:
                pass
            try:
                self._info[module_name].append((key, value))
            except KeyError:
                self._info[module_name] = [(key, value)]

    def balance(self):
        provided_energy = np.sum(self._info['provided_energy'])
        absorbed_energy = np.sum(self._info['absorbed_energy'])
        return provided_energy, absorbed_energy, self._reward

    def output(self):
        _info = {k: v for k, v in self._info.items() if k not in ('absorbed_energy', 'provided_energy')}
        return self._obs, self._reward, self._done, _info

    @property
    def obs(self):
        return self._obs

    @property
    def reward(self):
        return self._reward

    @property
    def done(self):
        return self._done

    @property
    def info(self):
        return self._info