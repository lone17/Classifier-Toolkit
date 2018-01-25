import csv
import sys

with open(sys.argv[1], 'r') as f :
    reader = csv.reader(f, delimiter=',')
    reader.__next__()
    currentFileName = None
    currenFile = None
    writer = None
    # print(','.join(reader.__next__()[1:]))
    for row in reader :
        # print(row[0])
        if row[0] != currentFileName :
            if currentFileName != None : currenFile.close()
            currenFile = open(row[0] + '.csv', 'a')
            writer = csv.writer(currenFile, delimiter=',')
        writer.writerow(row[1:])
