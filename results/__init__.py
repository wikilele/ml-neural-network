import os
import json
import matplotlib.pyplot as plt 

result_index = 0
grid_search_results = {}
TASK = ""

def set_task(task_name):
    global TASK
    if task_name not in ['monks1','monks2','monks3']:
        raise TypeError

    TASK = task_name
    if not os.path.isdir('results/plots/' + TASK):
            os.mkdir('results/plots/' + TASK)
    if not os.path.isdir('results/' + TASK):
            os.mkdir('results/' + TASK)

def save_plot(plt, mtype): # type can be mse, acc, mee
    global result_index
    path = 'results/plots/' + TASK + '/'+ mtype + str(result_index) + '.png'
    result_index +=1
    plt.savefig(path)
    plt.clf()
    return path

def add_result(mse, params, metrics_values, path):
        grid_search_results[mse] = {'metrics': metrics_values, 'params' : params, 'plotpath': path}
    
def save_results():
    with open('results/' + TASK + '/grid_results.json','w+') as f:
        f.write(json.dumps(grid_search_results,indent=4, sort_keys=True))