import json
import os

from trainAndFitLinear import train_model
import random
import numpy as np

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


#######################
a = 0.001  # GradNorm Weight
b = 0.000001  # Prediction Weight
g = 0.001  # Scale for Phi
lr = 0.01  # Learning Rate
#######################

a = np.linspace(0.0001, 0.01, 7)  # 0.00257
b = np.linspace(0, 0.0000001, 2)  # 1e-7
g = np.linspace(0.0001, 0.01, 7)  # 0.005
lr = np.geomspace(1e-5, 1e-2, 6)  # unknown

spaces = [a, b, g, lr]
ispaces = list(enumerate(spaces))
hypers = cartesian_product(a, b, g, lr)

seed_freq = 10
duration = 700001


history = dict()


def updateParams(new_params, new_perf, best_params, best_perf):
    print('RMS:', new_perf)
    print('Best RMS: ', min(new_perf))
    print('Avg RMS: ', sum(new_perf) / len(new_perf))

    # Evaluation method
    # Min RMS prediction error after RMS error reaches maximum
    max_idx = np.argmin(new_perf)
    max_idx_best = np.argmin(best_perf)

    if max_idx < 100000:
        return best_params, best_perf

    if min(new_perf[max_idx:]) < min(best_perf[max_idx_best:]) and min(new_perf) > 0:
        print('New Best!')
        print(100 * min(best_perf[max_idx_best:]) / min(new_perf[max_idx:]) - 100, "% improvement")
        return new_params, new_perf
    else:
        return best_params, best_perf


def params_to_string(params):
    return 'alpha:{} beta:{} gamma:{} lr:{}'.format(*params)


# Load json for sweep index
filename = './sweep_info.json'
if os.path.exists(filename):
    with open(filename, 'r') as f:
        folder_info = json.load(f)
else:
    folder_info = {'sweep_number': 0}

sweep_number = folder_info['sweep_number']
folder_info['sweep_number'] += 1

with open(filename, 'w') as f:
    json.dump(folder_info, f)

best_params = [0.00257, 1e-7, 0.005, 0.001]
best_ratio, best_perf = train_model(best_params[0], best_params[1], best_params[2], best_params[3], duration,
                                    "sweep_" + str(sweep_number))
history[params_to_string(best_params)] = (best_perf, best_ratio)

for i in range(1000):
    try:
        if i % seed_freq:
            # Re-seed search space, update if better
            new_params = random.choice(hypers)
        else:
            # Permute current hyper-params and update if better
            new_params = best_params
            idx, space = random.choice(ispaces)
            new_params[idx] = random.choice(space)

        print('Training Model: ', new_params)
        new_ratio, new_perf = train_model(new_params[0], new_params[1], new_params[2], new_params[3], duration, "sweep_" + str(sweep_number))

        # Save results
        history[params_to_string(new_params)] = (new_perf, new_ratio)

        # Update best params
        best_params, best_perf = updateParams(new_params, new_perf, best_params, best_perf)

        history['best_params'] = list(best_params)
        history['best_perf'] = best_perf

        with open('./sweep_results{}.json'.format(sweep_number), 'w') as f:
            json.dump(history, f, sort_keys=True, indent=4)

    except KeyboardInterrupt:
        exit(-1)

    except Exception as e:
        print(e)
        continue

with open('./sweep_results{}.json'.format(sweep_number), 'w') as f:
    json.dump(history, f, sort_keys=True, indent=4)
