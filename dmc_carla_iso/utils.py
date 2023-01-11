import os
from typing import Iterable

import cv2
import numpy as np
# import plotly
import torch
# from plotly.graph_objs import Scatter
# from plotly.graph_objs.scatter import Line
from torch.nn import Module
from torch.nn import functional as F


# # Plots min, max and mean + standard deviation bars of a population over time
# def lineplot(xs, ys_population, title, path='', xaxis='episode'):
#   max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

#   if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
#     ys = np.asarray(ys_population, dtype=np.float32)
#     ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
#     ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

#     trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
#     trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
#     trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
#     trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
#     trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
#     trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
#     data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
#   else:
#     data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
#   plotly.offline.plot({
#     'data': data,
#     'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
#   }, filename=os.path.join(path, title + '.html'), auto_open=False)


# def write_video(frames, title, path=''):
#   frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
#   _, H, W, _ = frames.shape
#   writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
#   for frame in frames:
#     writer.write(frame)
#   writer.release()


# def cal_returns(reward, value, bootstrap, pcont, lambda_):
#   """
#   Calculate the target value, following equation (5-6) in Dreamer
#   :param reward, value: imagined rewards and values, dim=[horizon, (chuck-1)*batch, reward/value_shape]
#   :param bootstrap: the last predicted value, dim=[(chuck-1)*batch, 1(value_dim)]
#   :param pcont: gamma
#   :param lambda_: lambda
#   :return: the target value, dim=[horizon, (chuck-1)*batch, value_shape]
#   """
#   assert list(reward.shape) == list(value.shape), "The shape of reward and value should be similar"
#   if isinstance(pcont, (int, float)):
#     pcont = pcont * torch.ones_like(reward)

#   next_value = torch.cat((value[1:], bootstrap[None]), 0)  # bootstrap[None] is used to extend additional dim
#   inputs = reward + pcont * next_value * (1 - lambda_)  # dim=[horizon, (chuck-1)*B, 1]
#   outputs = []
#   last = bootstrap

#   for t in reversed(range(reward.shape[0])): # for t in horizon
#     inp = inputs[t]
#     last = inp + pcont[t] * lambda_ * last
#     outputs.append(last)
#   returns = torch.flip(torch.stack(outputs), [0])
#   return returns


class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]
