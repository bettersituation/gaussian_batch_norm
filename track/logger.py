import csv


class CsvWriter:
    def __init__(self, fn):
        self.file = open(fn, 'a')
        self.csv = csv.writer(self.file)

    def writerow(self, contents):
        self.csv.writerow(contents)

    def close(self):
        self.file.close()
