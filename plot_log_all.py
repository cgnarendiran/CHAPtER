#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os.path
import re
import argparse
from os import listdir
from os.path import isfile, join

matplotlib.rcParams.update({'font.size': 22})
plt.style.use('seaborn-dark')

PLOT_DIR = 'plots'

def extract_test_from_log(log_path, x_label):
  assert(x_label == "Step" or x_label == "Episode")
  reg = re.compile(f"^{x_label} \d*$") # Needed to avoid issues with other lines starting with 'Episode'

  with open(log_path) as log:
    lines = log.readlines()
    lines = list(filter(lambda x: '(test) Average reward' in x or reg.match(x.split(' - ')[1]) is not None, lines))
    # print(lines)
    
    indices = list(filter(lambda x: '(test) Average reward' in x[1], enumerate(lines)))
    x = map(lambda x: int(lines[x[0] - 1].split(' - ')[1].split(' ')[1].rstrip()), indices)
    y = map(lambda x: float(lines[x[0]].split(' - ')[1].split(': ')[1].rstrip()), indices)

    return list(x), list(y)

def extract_test_std_from_log(log_path, x_label):
  assert(x_label == "Step" or x_label == "Episode")
  reg = re.compile(f"^{x_label} \d*$") # Needed to avoid issues with other lines starting with 'Episode'

  with open(log_path) as log:
    lines = log.readlines()
    lines = list(filter(lambda x: 'Standard deviation' in x or reg.match(x.split(' - ')[1]) is not None, lines))
    # print(lines)
    
    indices = list(filter(lambda x: 'Standard deviation' in x[1], enumerate(lines)))
    x = map(lambda x: int(lines[x[0] - 1].split(' - ')[1].split(' ')[1].rstrip()), indices)
    y = map(lambda x: float(lines[x[0]].split(' - ')[1].split(': ')[1].rstrip()), indices)

    return list(x), list(y)

def extract_train_from_log(log_path, x_label):
  assert(x_label == "Step" or x_label == "Episode")
  reg = re.compile(f"^{x_label} \d*$")

  with open(log_path) as log:
    lines = log.readlines()
    lines = list(filter(lambda x: 'Episode reward mean' in x or reg.match(x.split(' - ')[1]) is not None, lines))
    # print(lines)
    indices = list(filter(lambda x: 'Episode reward mean' in x[1], enumerate(lines)))
    x = map(lambda x: int(lines[x[0] - 1].split(' - ')[1].split(' ')[1].rstrip()), indices)
    y = map(lambda x: float(lines[x[0]].split(' - ')[1].split(': ')[1].rstrip()), indices)

    return list(x), list(y)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Log Parser')
  parser.add_argument('--game', type=str, default='lunar_sparse')
  parser.add_argument('--log_rer', type=str, default='logs/rer/')
  parser.add_argument('--log_per', type=str, default='logs/per/')
  parser.add_argument('--log_her', type=str, default='logs/her/')
  parser.add_argument('--name', type=str, default='_')

  parser.add_argument('--xmin', type=int, default=0)
  parser.add_argument('--xmax', type=int)
  parser.add_argument('--ymin', type=int, default=0)
  parser.add_argument('--ymax', type=int)
  parser.add_argument('--use-eps', action="store_true", default=False)
  parser.add_argument('--title', type=str)

  args = parser.parse_args()

  x_label = "Episode"  # if args.use_eps else "Step"
  save_file_name = f'{PLOT_DIR}/{args.name}_{args.game}.png'

  log_dirs = [args.log_rer, args.log_per, args.log_her]
  replay_colors = ['gray', 'crimson', 'dodgerblue']
  replay_labels = ['rer', 'per', 'her']

  plt.figure(figsize=(16, 12), dpi=100)

  for i, log_dir in enumerate(log_dirs):
      log_dir = log_dir.replace('logs/', 'logs/' + args.game + '/')
      filenames = [join(log_dir, f) for f in listdir(log_dir) if isfile(join(log_dir, f))]
      mean_test_x, mean_test_y, mean_test_error = [], [], []
      for name in filenames:
        test_x, test_y, train_x, train_y, test_error = ([], [], [], [], [])
        test_max = 0
        train_max = 0
        name = name.strip()
        a, b = extract_test_from_log(name, x_label)
        test_x += list(map(lambda x: x + test_max, a))
        test_y += b
        a, b = extract_train_from_log(name, x_label)
        train_x += list(map(lambda x: x + train_max, a))
        train_y += b
        _, a = extract_test_std_from_log(name, x_label)
        test_error += a
        test_max = max(test_x)
        train_max = max(train_x)
        mean_test_x.append(test_x)
        mean_test_y.append(test_y)
        mean_test_error.append(test_error)

      if filenames:
          mean_test_x = np.arange(0, 1000, 50)[:-1]

          for m in range(len(mean_test_y)):
              x = mean_test_x
              y = mean_test_y[m]
              test_plot = plt.plot(x, y, color=replay_colors[i], linestyle='dashed', alpha=0.4,
                                   ms=25.0)  # + ' 10 episode average test reward')

          std_test_seed = np.std(mean_test_y, axis=0)
          # mean_test_x = np.mean(mean_test_x, axis=0)
          mean_test_y = np.mean(mean_test_y, axis=0)

          # mean_test_x = np.arange(0, 1000, 50)[:-1]

          mean_test_error = np.mean(mean_test_error, axis=0)

          plt.fill_between(mean_test_x, np.array(mean_test_y) - np.array(std_test_seed),
                           np.array(mean_test_y) + np.array(std_test_seed), color=replay_colors[i], alpha=0.25)
          test_plot = plt.plot(mean_test_x, mean_test_y, color=replay_colors[i],
                               ms=25.0, label=replay_labels[i])  # + ' 10 episode average test reward')
      # train_plot = plt.plot(train_x, train_y, color=replay_colors[i],
      #                       linestyle='dashed',  ms=25.0, label=replay_labels[i] + ' 10 episode average train reward')


  plt.xlabel(x_label)
  plt.ylabel('Mean test reward avg over seeds')
  plt.title(args.game)
  plt.legend()
  plt.savefig(save_file_name)
  plt.close()

  print(f"Saved to {save_file_name}")