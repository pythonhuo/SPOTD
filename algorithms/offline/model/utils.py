import numpy as np
import torch

import os
import random
import imageio
import gym
from tqdm import trange
import pickle


def print_banner(s, separator="-", num_star=60):
    print(separator * num_star, flush=True)
    print(s, flush=True)
    print(separator * num_star, flush=True)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_finetune(self, dataset):
        self.ptr = dataset['observations'].shape[0]
        self.size = dataset['observations'].shape[0]
        self.state[:self.ptr] = dataset['observations']
        self.action[:self.ptr] = dataset['actions']
        self.next_state[:self.ptr] = dataset['next_observations']
        self.reward[:self.ptr] = dataset['rewards'].reshape(-1, 1)
        self.not_done[:self.ptr] = 1. - dataset['terminals'].reshape(-1, 1)

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def clip_to_eps(self, eps=1e-5):
        lim = 1 - eps
        self.action = np.clip(self.action, -lim, lim)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")


class VideoRecorder(object):
    def __init__(self, dir_name, height=512, width=512, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                # camera_id=self.camera_id
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np


def print_banner(s, separator="-", num_star=60):
    print(separator * num_star, flush=True)
    print(s, flush=True)
    print(separator * num_star, flush=True)


class Progress:

    def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self._speed_update_freq = speed_update_freq

        self._step = 0
        self._prev_line = '\033[F'
        self._clear_line = ' ' * self.line_width

        self._pbar_size = self.ncol * self.max_length
        self._complete_pbar = '#' * self._pbar_size
        self._incomplete_pbar = ' ' * self._pbar_size

        self.lines = ['']
        self.fraction = '{} / {}'.format(0, self.total)

        self.resume()

    def update(self, description, n=1):
        self._step += n
        if self._step % self._speed_update_freq == 0:
            self._time0 = time.time()
            self._step0 = self._step
        self.set_description(description)

    def resume(self):
        self._skip_lines = 1
        print('\n', end='')
        self._time0 = time.time()
        self._step0 = self._step

    def pause(self):
        self._clear()
        self._skip_lines = 1

    def set_description(self, params=[]):

        if type(params) == dict:
            params = sorted([
                (key, val)
                for key, val in params.items()
            ])

        ############
        # Position #
        ############
        self._clear()

        ###########
        # Percent #
        ###########
        percent, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

        #########
        # Speed #
        #########
        speed = self._format_speed(self._step)

        ##########
        # Params #
        ##########
        num_params = len(params)
        nrow = math.ceil(num_params / self.ncol)
        params_split = self._chunk(params, self.ncol)
        params_string, lines = self._format(params_split)
        self.lines = lines

        description = '{} | {}{}'.format(percent, speed, params_string)
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr):
        self.lines.append(descr)

    def _clear(self):
        position = self._prev_line * self._skip_lines
        empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
        print(position, end='')
        print(empty)
        print(position, end='')

    def _format_percent(self, n, total):
        if total:
            percent = n / float(total)

            complete_entries = int(percent * self._pbar_size)
            incomplete_entries = self._pbar_size - complete_entries

            pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
            fraction = '{} / {}'.format(n, total)
            string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
        else:
            fraction = '{}'.format(n)
            string = '{} iterations'.format(n)
        return string, fraction

    def _format_speed(self, n):
        num_steps = n - self._step0
        t = time.time() - self._time0
        speed = num_steps / t
        string = '{:.1f} Hz'.format(speed)
        if num_steps > 0:
            self._speed = string
        return string

    def _chunk(self, l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def _format(self, chunks):
        lines = [self._format_chunk(chunk) for chunk in chunks]
        lines.insert(0, '')
        padding = '\n' + ' ' * self.indent
        string = padding.join(lines)
        return string, lines

    def _format_chunk(self, chunk):
        line = ' | '.join([self._format_param(param) for param in chunk])
        return line

    def _format_param(self, param):
        k, v = param
        return '{} : {}'.format(k, v)[:self.max_length]

    def stamp(self):
        if self.lines != ['']:
            params = ' | '.join(self.lines)
            string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
            self._clear()
            print(string, end='\n')
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self):
        self.pause()


class Silent:

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        return lambda *args: None


class EarlyStopping(object):
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        else:
            self.counter = 0
        return False
