import matplotlib.pyplot as plt
from mpltools import style
import csv
import sys
import os
import glob
import re

def get_avg_val(val_str):
    return float(val_str.split('\xb1')[0])

col_fcns = {'bench_fcn': int,
     'num_swarms': int,
     'num_sparticles': int,
     'max_iters': int,
     'max_ga_init_iters': int,
     'omega': float,
     'omega_decay': float,
     'c1': float,
     'c2': float,
     'exchange_iters': int,
     'num_exchange': int,
     'max_axis_val': float,
     'max_vel': float,
     'num_dims': int,
     'num_reps': int,
     'm': int,
     'Avg Execution Time': get_avg_val,
     'Avg Fitness': get_avg_val,
        }

def row_to_data(row, headers):
    params = {}
    samples = []
    for col_index in range(len(row)):
        col_title = headers[col_index]
        if re.match(r'^Sample \d+$', col_title):
            samples.append(get_avg_val(row[col_index]))
        else:
            if col_title in col_fcns:
                row[col_index] = col_fcns[col_title](row[col_index])
            params[col_title] = row[col_index]

    return params, samples

def main():
    if len(sys.argv) != 2:
        print 'python plot.py <alg_name>'
        exit()

    plt.rcParams['figure.max_open_warning'] = 30

    alg_name = sys.argv[1]

    alg_dir = 'C:/Users/Wayne/Documents/School/Thesis-work/experiments/thesis/%s/' % (alg_name)
    csv_filenames = get_csv_filenames(alg_dir)

    for item in csv_filenames:
        plot(item)

def get_csv_filenames(cur_dir):
    filenames = []

    contents = os.listdir(cur_dir)
    for item in contents:
        path = '%s%s' % (cur_dir, item)
        if os.path.isdir(path) and item != 'test':
            path += '/'
            filenames.extend(get_csv_filenames(path))
        elif path.lower().endswith('.csv'):
            filenames.append(path)

    return filenames
    
def plot(src_filename):
    #global num_figs
    
    src_dir = os.path.dirname(src_filename) + '/'
    
    #remove any existing png files in the directory
    existing_pngs = glob.glob('%s*.png' % (src_dir))
    for png_filename in existing_pngs:
        os.remove(png_filename)
    
    csv_file = open(src_filename, 'rb')
    reader = csv.reader(csv_file, delimiter=',')
    rows = list(reader)
    csv_file.close()

    if len(rows) == 0:
        print 'No data in csv file.'
        exit()

    headers = rows[0]

    style.use('ggplot')

    print '\nGenerating graphs for %s:' % (src_filename)
    for i in range(1, len(rows)):
        print 'Graph %d of %d' % (i, len(rows) - 1)
        
        params, samples = row_to_data(rows[i], headers)
        max_iters = params['max_iters']

        #for ga
        if 'max_ga_init_iters' in params:
            max_iters += params['max_ga_init_iters']
        
        sample_interval = max_iters / len(samples) + (1 if max_iters % len(samples) else 0)
        x_ticks = range(0, max_iters, sample_interval)

        #append final fitness value
        x_ticks.append(max_iters)
        samples.append(params['Avg Fitness'])
        
        fig = plt.figure(i)
        fig.clear() #clear figure in case this function has already been called for this figure
        fig.suptitle('F%d' % (params['bench_fcn']))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Iteration Index')
        ax.set_ylabel('Best Fitness')
        plt.axis([0, max_iters, 0, samples[len(samples) / 2] * 3])
        #plt.axis([0, max_iters, 0, samples[0]])
        
        plt.plot(x_ticks, samples, antialiased=True)
        plt.savefig('%sfig-%d.png' % (src_dir, i), dpi=300)

    #plt.show()

main()
