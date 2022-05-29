import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings
from agents.navigation.carla_env_dream import CarlaEnv

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import torch
from torch import distributions as torchd
from torch import nn

import exploration as expl
import models
import tools
import wrappers

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):

  def __init__(self, config, logger, dataset):
    super(Dreamer, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = {}
    self._step = count_steps(config.traindir)
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = dataset
    self._wm = models.WorldModel(self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._wm, config.behavior_stop_grad)
    reward = lambda f, s, a: self._wm.heads['reward'](f).mean
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()
    self.step = 0
    self.recon = []
    self.gen_action = []
    self.gen_free = []
    self.mask_action = []
    self.mask_free = []
    self.mask_3 = []
    self.back = []
    self.obs = []

  def __call__(self, obs, reset, state=None, reward=None, training=True):
    step = self._step
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = 1 - reset
      for key in state[0].keys():
        for i in range(state[0][key].shape[0]):
          state[0][key][i] *= mask[i]
      for i in range(len(state[1])):
        state[1][i] *= mask[i]
    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      for _ in range(steps):
        self._train(next(self._dataset))
      if self._should_log(step):
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        openl = self._wm.video_pred(next(self._dataset))
        self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)

    policy_output, state = self._policy(obs, state, training)

    if training:
      self._step += len(reset)
      self._logger.step = self._config.action_repeat * self._step
    return policy_output, state

  def _policy(self, obs, state, training):
    self.step += 1
    # obs = torch.load(f'fig/obs_{self.step}.pkl')
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
    else:
      latent, action = state
    embed = self._wm.encoder(self._wm.preprocess(obs))
    latent, _ = self._wm.dynamics.obs_step(
        latent, action, embed, self._config.collect_dyn_sample)

    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']

    self._task_behavior.init_men()
    latent_free = latent.copy()
    for _ in range(self._config.window):
      stoch_free = latent_free['stoch_free']
      deter_free = latent_free['deter_free']
      free_feat = torch.cat([stoch_free, deter_free], -1)
      self._task_behavior.men.append(free_feat)       
      latent_free = self._wm.dynamics.img_step(latent_free, None, sample=self._config.imag_sample, only_free=True)
    free_atten = torch.stack(self._task_behavior.men, dim=1)
    feat = self._wm.dynamics.get_feat(latent, free_atten, self._task_behavior.attention)
    if not training:
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
    elif self._should_expl(self._step):
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
    logprob = actor.log_prob(action)
    latent = {k: v.detach()  for k, v in latent.items()}
    action = action.detach()
    if self._config.actor_dist == 'onehot_gumble':
      action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  def _train(self, data):
    metrics = {}
    post, context, mets = self._wm._train(data)
    metrics.update(mets)
    start = post
    if self._config.pred_discount:  # Last step could be terminal.
      start = {k: v[:, :-1] for k, v in post.items()}
      context = {k: v[:, :-1] for k, v in context.items()}
    reward = lambda f, s, a: self._wm.heads['reward'](
        self._wm.dynamics.get_feat(s)).mode()
    metrics.update(self._task_behavior._train(start, reward)[-1])
    if self._config.expl_behavior != 'greedy':
      if self._config.pred_discount:
        data = {k: v[:, :-1] for k, v in data.items()}
      mets = self._expl_behavior.train(start, context, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    for name, value in metrics.items():
      if not name in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)


def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
  generator = tools.sample_episodes(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tools.from_generator(generator, config.batch_size)
  return dataset


def make_env(config, logger, mode, train_eps, eval_eps):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, config.size,
        grayscale=config.grayscale,
        life_done=False and ('train' in mode),
        sticky_actions=True,
        all_actions=True)
    env = wrappers.OneHotAction(env)
  elif suite == 'dmlab':
    env = wrappers.DeepMindLabyrinth(
        task,
        mode if 'train' in mode else 'test',
        config.action_repeat)
    env = wrappers.OneHotAction(env)
  elif suite == 'carla':
    env = CarlaEnv(
            render_display=False,  # for local debugging only
            display_text=False,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=config.image_size,
            max_episode_steps=1000,
            frame_skip=config.action_repeat,
            is_other_cars=True,
            port=2000
        )
    env_eval = env
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit)
  env_eval = wrappers.TimeLimit(env_eval, config.time_limit)
  env = wrappers.SelectAction(env, key='action')
  env_eval = wrappers.SelectAction(env_eval, key='action')
  mode = 'train'
  callbacks = [functools.partial(
        process_episode, config, logger, mode, train_eps, eval_eps)]
  env = wrappers.CollectDataset(env, callbacks)
  mode = 'eval'
  callbacks_eval = [functools.partial(
        process_episode, config, logger, mode, train_eps, eval_eps)]
  env_eval = wrappers.CollectDataset(env_eval, callbacks_eval)
  env = wrappers.RewardObs(env)
  env_eval = wrappers.RewardObs(env_eval)
  # print(env==env_eval)
  return env, env_eval


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  length = len(episode['reward']) - 1
  if length >= 50:
    filename = tools.save_episodes(directory, [episode])[0]
    
    if mode == 'eval':
      cache.clear()
    if mode == 'train' and config.dataset_size:
      total = 0
      for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if total <= config.dataset_size - length:
          total += len(ep['reward']) - 1
        else:
          del cache[key]
      logger.scalar('dataset_size', total + length)
    cache[str(filename)] = episode
    logger.scalar(f'{mode}_episodes', len(cache))

  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()


def main(config):
  logdir = pathlib.Path(config.logdir).expanduser()
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat
  config.act = getattr(torch.nn, config.act)

  print('Logdir', logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  config.traindir.mkdir(parents=True, exist_ok=True)
  config.evaldir.mkdir(parents=True, exist_ok=True)
  step = count_steps(config.traindir)
  logger = tools.Logger(logdir, config.action_repeat * step)

  print('Create envs.')
  if config.offline_traindir:
    directory = config.offline_traindir.format(**vars(config))
  else:
    directory = config.traindir
  train_eps = tools.load_episodes(directory, limit=config.dataset_size)
  if config.offline_evaldir:
    directory = config.offline_evaldir.format(**vars(config))
  else:
    directory = config.evaldir
  eval_eps = tools.load_episodes(directory, limit=1)
  make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
  envs_list = [make('train') for _ in range(config.envs)]
  train_envs = [envs_list[0][0]]
  eval_envs = [envs_list[0][1]]
  acts = train_envs[0].action_space
  config.num_actions = acts.shape[0]

  if not config.offline_traindir:
    prefill = max(0, config.prefill - count_steps(config.traindir))
    print(f'Prefill dataset ({prefill} steps).')

    random_actor = torchd.independent.Independent(
          torchd.uniform.Uniform(torch.Tensor([acts.low.min(), acts.low.min()])[None],
                                 torch.Tensor([acts.high.max(), acts.high.max()])[None]), 1)
    def random_agent(o, d, s, r):
      action = random_actor.sample()
      logprob = random_actor.log_prob(action)
      return {'action': action, 'logprob': logprob}, None
    tools.simulate(random_agent, train_envs, prefill)
    tools.simulate(random_agent, eval_envs, episodes=1)
    logger.step = config.action_repeat * count_steps(config.traindir)

  train_dataset = make_dataset(train_eps, config)
  eval_dataset = make_dataset(eval_eps, config)
  agent = Dreamer(config, logger, train_dataset).to(config.device)
  agent.requires_grad_(requires_grad=False)
  agent.load_state_dict(torch.load(logdir / 'model_106555.pt'))
  agent._should_pretrain._once = False

  torch.backends.cudnn.enabled = False

  state = None
  i = 0
  while i < 5:
    logger.write()
    print('Start evaluation.')
    agent.eval()
    eval_policy = functools.partial(agent, training=False)
    tools.simulate(eval_policy, eval_envs, episodes=1)

    video_pred = agent._wm.video_pred(next(eval_dataset))
    logger.video('eval_openl', to_np(video_pred))
    i += 1
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  main(parser.parse_args(remaining))

