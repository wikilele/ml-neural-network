import os
import matplotlib.pyplot as plt 

result_index = 0
gridsearch_results_header = ""
gridsearch_results_body = ""
TASK = None


def set_task(task_name):
    global TASK
    if task_name not in ['monks-1','monks-2','monks-3']:
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

def add_result_header(*args):
    global gridsearch_results_header
    for e in args:
        gridsearch_results_header += e + ", "
    gridsearch_results_header += '\n'

def add_result(*args):
    global gridsearch_results_body
    for e in args:  
        gridsearch_results_body += str(e) + ", "
    gridsearch_results_body += '\n'
    #grid_search_results[mse] = {'metrics': metrics_values, 'params' : params, 'plotpath': path}
    
def save_results():
    with open('results/' + TASK + '/grid_results.csv','w+') as f:
        f.write(gridsearch_results_header)
        f.write(gridsearch_results_body)