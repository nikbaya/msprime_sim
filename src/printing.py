def sec_to_str(t):
    # Convert seconds to days:hours:minutes:seconds
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f

def _remove_dtype(x):
    # Removes dtype: float64 and dtype: int64 from pandas printouts
    x = str(x)
    x = x.replace('\ndtype: int64', '')
    x = x.replace('\ndtype: float64', '')
    return x

class Logger(object):
    # Lightweight logging.

    def __init__(self, fh):
        self.log_fh = open(fh, 'wb')

    def log(self, msg):
        # Print to log file and stdout with a single command.

        print >>self.log_fh, msg
        print msg