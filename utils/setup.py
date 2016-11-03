import sys
import re
import os
import glob

conf_dir = 'C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/mpso-gpu/config/'
batch_dir = 'C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/mpso-gpu/utils/'

def split_file(alg_str, chunk_size):
    conf_name = 'config_%s' % (alg_str)
    in_file = open('%s%s/%s.txt' % (conf_dir, alg_str, conf_name), 'rb')

    text_files = glob.glob('%s%s/*.txt' % (conf_dir, alg_str))
    for txt in text_files:
        if re.match(r'^%s-\d+\.txt$' % (conf_name), os.path.basename(txt)):
            os.remove(txt)
    
    in_lines = in_file.readlines()
    in_file.close()
    headers = in_lines[0:2]

    configs = []

    in_lines = filter(lambda line: not (line.startswith('#') or line.strip() == ''), in_lines[2:])

    i = 0
    while i < len(in_lines):
        chunk = []
        chunk.extend(headers)
        j = 0
        while i < len(in_lines) and j < chunk_size:
            chunk.append(in_lines[i])
            i += 1
            j += 1
        configs.append(chunk)

    for i in range(len(configs)):
        out_file = open('%s%s/%s-%d.txt' % (conf_dir, alg_str, conf_name, i), 'wb')
        for line in configs[i]:
            out_file.write(line)
        out_file.close()

    in_file.close()

    return len(configs)

def gen_batch_file(alg_str, num_configs):
    batch_file = open('%srun.bat' % (batch_dir), 'wb')

    batch_file.write('echo off\r\n')
    #batch_file.write('call update.bat\r\n')
    batch_file.write('cd "C:\\Users\Wayne\\Documents\Visual Studio 2010\\Projects\mpso-gpu\Debug"\r\n')

    op = '>'
    if os.path.exists('C:\\Users\\Wayne\\Documents\\Visual Studio 2010\\Projects\\mpso-gpu\\%s.csv' % (alg_str)):
        op = '>>'
        
    batch_file.write('echo "" %s "C:\\Users\\Wayne\\Documents\\Visual Studio 2010\\Projects\\mpso-gpu\\%s.csv"\r\n' % (op, alg_str))
    batch_file.write('for /L %%%%i in (0,1,%d) do (\r\n' % (num_configs - 1))
    batch_file.write('"C:\\Users\\Wayne\\Documents\\Visual Studio 2010\\Projects\\mpso-gpu\\Debug\\mpso-gpu.exe" config\\%s\\config_%s-%%%%i.txt >> "C:\\Users\\Wayne\\Documents\\Visual Studio 2010\\Projects\\mpso-gpu\\%s.csv"\r\n' % (alg_str, alg_str, alg_str))
    batch_file.write(')\r\n')
    batch_file.write('python "C:\\Users\\Wayne\\Documents\\Visual Studio 2010\\Projects\\mpso-gpu\\mpso-gpu\\utils\\cleanup_headers.py" %s\r\n' % (alg_str))
    
    batch_file.close()

def main():
    if len(sys.argv) != 3:
        print 'Usage: python splitter.py <config_name> <lines_per_subconfig>'
        exit(0)

    alg_str = sys.argv[1]
    chunk_size = int(sys.argv[2])

    num_configs = split_file(alg_str, chunk_size)

    gen_batch_file(alg_str, num_configs)

main()
