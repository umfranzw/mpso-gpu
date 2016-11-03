import sys
import csv
import re

if len(sys.argv) != 2:
    print 'Usage: python cleanup_headers.py <alg_name>'
    exit()

csv_filename = 'C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/%s.csv' % (sys.argv[1])

csv_file = open(csv_filename, 'rb')
reader = csv.reader(csv_file, delimiter=',')
rows = list(reader)
csv_file.close()

csv_file = open(csv_filename, 'wb')
writer = csv.writer(csv_file, delimiter=',')
wrote_header = False

for i in range(len(rows)):
    is_good = len(rows[i]) > 0 and rows[i][0].strip() != '' and rows[i][0].strip() != 'Timestamp'

    is_header = is_good and rows[i][0].strip() == 'bench_fcn'
    
    if not wrote_header and is_header:
        wrote_header = True
    elif is_header:
        is_good = False
        
    if is_good:
        writer.writerow(rows[i])
