
# coding: utf-8

# In[ ]:

from __future__ import print_function
import sys
import re
from time import clock


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go %(timeleft)f s'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0
        self.startTime = clock()

    def __call__(self,current=None):
        '''Assumes current is zero based index completed, i.e. current=0 means 1 iteration completed'''
        if current is not None:
            self.current = current + 1
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        remainingTime = float(remaining) * (clock() - self.startTime)/float(self.current)
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining,
            'timeleft':remainingTime
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
        print('Completed in {} seconds.'.format(clock() - self.startTime), file=self.output, end='')
        print('', file=self.output)
if __name__ == '__main__':
    from time import sleep

    progress = ProgressBar(80, fmt=ProgressBar.FULL)

    for x in xrange(progress.total):
        progress(x)
        sleep(0.1)
    progress.done()

