
# coding: utf-8

# In[ ]:

from sys import stdout

class Logger(object):
    def __init__(self,logFile=None,standardOut = True):
        if standardOut:
            self.file = [stdout]
        else:
            self.file = []
        if logFile is not None:
            try:
                file = open(logFile,"w")
                self.file.append(file)
            except:
                print("Failed to open log file {0}".format(logFile))
        if len(self.file) == 0:
            print("No logger files!")
            exit(1)
        #print("Using log files: {}".format(self.file))
    def __exit__(self):
        for file in self.file:
            try:
                file.close()
            except:
                pass
    def log(self,message,endLine=True):
        for f in self.file:
            if endLine:
                f.write("{0}\n".format(message))
            else:
                f.write("{0}".format(message))
            f.flush()

