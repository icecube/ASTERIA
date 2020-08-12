import numpy as np
import os
from datetime import datetime, timedelta
from ringbuffer import WindowBuffer

run_start = datetime(year=2020, month=2, day=19, hour=22, minute=0, second=6, microsecond=622426)
run_stop = datetime(year=2020, month=2, day=20, hour=6, minute=0, second=21, microsecond=881469)

class datetime_ns:

    # def __init__(self, year=None, month=None, day=None, hour=None, minute=None, second=None, nanosecond=None):
    def __init__(self, dt, nanosecond=None):
        self.datetime = dt
        if nanosecond is not None:
            self.ns = nanosecond
        else:
            self.ns = 0

    def __repr__(self):
        return '{0}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}.{6:09d}'.format(
            self.datetime.year, self.datetime.month, self.datetime.day,
            self.datetime.hour, self.datetime.minute, self.datetime.second, self.ns
        )

    def timedelta_ns(self, t_other):
        """
        Gets difference in time between self and other time t_other (datetime_ns) in nanoseconds.
        """
        delta = self.datetime - t_other.datetime
        return int(delta.total_seconds() * 1e9 + (self.ns - t_other.ns))


year_start = datetime_ns(datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0), nanosecond=0)
alert_time = datetime_ns(datetime(year=2020, month=2, day=19, hour=22, minute=36, second=6), nanosecond=288426276)


def utctime_to_datetime_ns(dt):
    """
    Converts UTC time element of hit to datetime_ns object
    """
    ns = int(dt % 1e10) // 10
    delta = timedelta(seconds=dt // 1e10)
    return datetime_ns(year_start.datetime+delta, ns)

def compute_dom_launch_time(hit):
    """
    Converts DOM clock element of hit to datetime_ns object
    """
    utctime = utctime_to_datetime_ns(hit.utc_time)
    ns = hit.dom_clk * 25
    delta = timedelta(seconds=ns // 1e9)
    ns = int(ns % 1e9)
    offset =timedelta(seconds = 0)
    if ns > utctime.ns: # Prevents negative ns element in datetime_ns, rolls back onto previous second
        offset = timedelta(seconds=1)
    return datetime_ns( utctime.datetime - delta - offset, abs(utctime.ns - ns))



class DeltaCompressedHit:

    def __init__(self, hit):
        self.hit_length = hit['hit_length']
        self.hit_type = hit['hit_type']
        self.dom_id = hit['dom_id']
        self.utc_time = hit['utc_time']
        self.byte_order = hit['byte_order']
        self.version = hit['version']
        self.pedestal = hit['pedestal']
        self.dom_clk = hit['dom_clk']
        self.word1 = hit['word1']
        self.word3 = hit['word3']

    @property
    def dom_id_str(self):
        return "{0:12x}".format(self.dom_id)

    # word1
    @property
    def hit_size(self):
        return self.word1 & 0x7ff

    @property
    def atwd_chip(self):
        return (self.word1 >> 11) & 0x1

    @property
    def atwd_size(self):
        return (self.word1 >> 12) & 0x03

    @property
    def atwd_available(self):
        return (self.word1 >> 14) & 0x1

    @property
    def fadc_available(self):
        return (self.word1 >> 15) & 0x1

    @property
    def lc(self):
        return (self.word1 >> 16) & 0x03

    @property
    def trigger_word(self):
        return (self.word1 >> 18) & 0xfff

    @property
    def min_bias(self):
        return (self.word1 >> 30) & 0x1

    @property
    def compression_flag(self):
        return (self.word1 >> 31) & 0x1

    # word3
    @property
    def post_peak_count(self):
        return self.word3 & 0x1FF

    @property
    def peak_count(self):
        return (self.word3 >> 9) & 0x1FF

    @property
    def pre_peak_count(self):
        return (self.word3 >> 18) & 0x1FF

    @property
    def peak_sample(self):
        return (self.word3 >> 27) & 0xF

    @property
    def peak_range(self):
        return (self.word3 >> 31) & 0x1


dtypes = [('hit_length', '>u4'),
          ('hit_type', '>u4'),
          ('dom_id', '>u8'),
          ('unused', '>u8'),
          ('utc_time', '>u8'),
          ('byte_order', '>u2'),
          ('version', '>u2'),
          ('pedestal', '>u2'),
          ('dom_clk', '>u8'),
          ('word1', '>u4'),
          ('word3', '>u4')]






def get_dom_ids(i_string=1):
    dtypes = [('str', 'f4'),
              ('i', 'f4'),
              ('x', 'f8'),
              ('y', 'f8'),
              ('z', 'f8'),
              ('mbid', 'S12'),
              ('type', 'S2'),
              ('effvol', 'f8')]

    table = np.genfromtxt('./data/full_dom_table.txt', dtype=dtypes)
    return table[table['str'] == i_string]['mbid']


def get_dom_launch_times(i_string):
    dtypes = [('hit_length', '>u4'),
              ('hit_type', '>u4'),
              ('dom_id', '>u8'),
              ('unused', '>u8'),
              ('utc_time', '>u8'),
              ('byte_order', '>u2'),
              ('version', '>u2'),
              ('pedestal', '>u2'),
              ('dom_clk', '>u8'),
              ('word1', '>u4'),
              ('word3', '>u4')]

    ids = get_dom_ids(i_string)
    launches = {}
    path = './data/raw/ichub01/HitSpool-35662.dat'
    filesize = os.path.getsize(path)
    for id in ids:
        launch_time_found = False
        offset = 0
        while not launch_time_found and offset < filesize:
            hit = np.fromfile(path, dtype=dtypes, count=1, offset=offset)[0]
            offset += hit['hit_length']
            if hit['dom_id'] == int(id, 16):
                print('Found Launch time for DOM {0}'.format(id.decode('ascii')))
                temp_hit = DeltaCompressedHit(hit)
                launch_time = compute_dom_launch_time(temp_hit)
                launches[id.decode('ascii')] = launch_time
                launch_time_found = True
    return launches


def convert_injection_time_to_utc(sn_trigger_time, hit_time):
    """
    Generates utc_time member /10 (To get in expected format multiple return by 10)
    """
    delta = timedelta(seconds=hit_time // 1e9)
    delta_ns = int(hit_time % 1e9)

    utc_time_ns = delta_ns + sn_trigger_time.ns
    if utc_time_ns < 0:
        delta = delta + timedelta(seconds=-1)
        utc_time_ns = int(1e9 + utc_time_ns)
    elif utc_time_ns >= 1e9:
        delta = delta + timedelta(seconds=1)
        utc_time_ns -= int(1e9)
    utc_time = sn_trigger_time.datetime + delta
    return datetime_ns(utc_time, utc_time_ns)


def convert_injection_time_to_domclk(launch_time, utc_hit_time):
    """
    Generates dom_clk member
    """
    delta = utc_hit_time.datetime - launch_time.datetime
    delta_ns = utc_hit_time.ns - launch_time.ns

    total_delta_ns = int(delta.total_seconds() * 1e9 + delta_ns)
    return total_delta_ns // 25



for item in get_dom_launch_times(1):
    print('ID: {0} \t t0: {1}'.format(item[0], item[1]))

dtypes = [('time', 'i'),
          ('mbid', 'U12')]
injection_hits = np.genfromtxt('./data/hits/ichub01/signal.dat', dtype=dtypes)

launch_times = get_dom_launch_times(1)




dtypes = [('hit_length', '>u4'),
          ('hit_type', '>u4'),
          ('dom_id', '>u8'),
          ('unused', '>u8'),
          ('utc_time', '>u8'),
          ('byte_order', '>u2'),
          ('version', '>u2'),
          ('pedestal', '>u2'),
          ('dom_clk', '>u8'),
          ('word1', '>u4'),
          ('word3', '>u4')]


def generate_word1():
    return 0


def generate_word3():
    return 0


ids = get_dom_ids(1)
launches = {}
path = './data/raw/ichub01/HitSpool-35662.dat'

file_list = [
    './data/raw/ichub01/HitSpool-35662.dat',
    './data/raw/ichub01/HitSpool-35663.dat',
    './data/raw/ichub01/HitSpool-35664.dat',
    './data/raw/ichub01/HitSpool-35665.dat',
    './data/raw/ichub01/HitSpool-35666.dat',
    './data/raw/ichub01/HitSpool-35667.dat',
    './data/raw/ichub01/HitSpool-35668.dat',
]

file_list = ['./data/raw/ichub01/HitSpool-35664.dat']

dtypes = [('str', 'f4'),
          ('i', 'f4'),
          ('x', 'f8'),
          ('y', 'f8'),
          ('z', 'f8'),
          ('mbid', 'S12'),
          ('type', 'S2'),
          ('effvol', 'f8')]
dom_table = np.genfromtxt('./data/full_dom_table.txt', dtype=dtypes)


def get_neighboring_doms(dom_id, table):
    dom = table[table['mbid'] == dom_id]
    string_doms = table[table['str']==dom['str']]
    if dom['i'] < 3:
        temp = string_doms[:dom['i']+2]
    elif dom['i'] > 58:
        temp = string_doms[dom['i']-2:]
    else:
        temp = string_doms[dom['i']-2:dom['i']+2]
    return temp['mbid'][temp['mbid'] != dom_id]

def get_dom_relative_location(this_id, other_id, table):
    this_dom = table[table['mbid'] == this_id]
    other_dom = table[table['mbid'] == other_id]
    if this_dom['i'] > other_dom['i']:
        return True  # This DOM is above Other DOM,
    elif this_dom['i'] < other_dom['i']:
        return False  # This DOM is below Other DOM


def get_hits_to_update(hit_buffer, injected_hit):
    #first update old hits
    # then update new hit
    dom_id_list = get_neighboring_doms(injected_hit['dom_id'], dom_table)
    idx_hits = hit_buffer.data['dom_id'].searchsorted(dom_id_list)
    for i, hit in enumerate(hit_buffer.data[idx_hits]):
        inj_is_above = get_dom_relative_location(injected_hit['dom_id'], hit['dom_id'], dom_table)
        hit_LC = hit['word1'] >> 16 & 0x03
        if inj_is_above and hit_LC == 1: # Old flag was from below, injected hit from above, will now be both
            hit_LC = 3
        elif inj_is_above and (hit_LC == 2 or hit_LC == 3): # No need to update if flag is already above
            pass
        elif not inj_is_above and hit_LC == 2:  # Old flag was from above, injected hit from below, now flag is both
            hit_LC = 3
        elif not inj_is_above and (hit_LC == 1 or hit_LC == 3):
            pass
        
offset = 0
buffer_2ms = WindowBuffer(80)
for injected_hit in injection_hits:
    n_iter = 0
    for path in file_list:
        filesize = os.path.getsize(path)
        while offset < filesize and n_iter < 600000:
            hit = np.fromfile(path, dtype=dtypes, count=1, offset=offset)[0]
            offset += hit['hit_length']
            utctime = convert_injection_time_to_utc(alert_time, injected_hit['time'])
            if utctime.timedelta_ns(year_start) * 10 < hit['utc_time']:
                print('INJECTED HIT ADDED')
                temp = np.zeros(1, dtype=dtypes)
                temp['hit_length'] = 54
                temp['hit_type'] = 3
                temp['dom_id'] = int(injected_hit['mbid'], 16)
                temp['utc_time'] = utctime.timedelta_ns(year_start) * 10
                temp['byte_order'] = 1
                temp['version'] = 2
                temp['pedestal'] = 1
                launch_time = launch_times[injected_hit['mbid']]
                temp['dom_clk'] = convert_injection_time_to_domclk(launch_time, utctime)
                temp['word1'] = generate_word1()
                temp['word3'] = generate_word3()
                # Get list of hits meeting the following conditions
                #  Within 1 or 2 Doms of triggering DOM on string
                #  Hit occurs within 1us of triggering hit
                # Generate hit status above, below or both
                #  Requires considering existing LC info

                buffer_2ms.append(temp)
                buffer_2ms.append(hit)
                break
            buffer_2ms.append(hit)
            n_iter += 1
        break
    print('Attempted {0} iterations'.format(n_iter))
    print('First UTC Time: {0}'.format(utctime.timedelta_ns(year_start) * 10))
    print('Final UTC Time: {0}'.format(hit['utc_time']))
    print('filesize {0}'.format(filesize))
    print('bytes read {0}'.format(offset))
    break





# for hit in injection_hits[:10]:
#     utctime =
#     launch_time = launch_times[hit['mbid']]
#     dom_clk = convert_injection_time_to_domclk(launch_time, utctime)



# 2000 iterations 43149300494887441
# 5000 iterations 43149301278593962

# offset = 0
# for i in range(10):
#     test = np.fromfile('./data/raw/ichub01/HitSpool-35662.dat', dtype=dtypes, count=1, offset=offset)[0]
#     offset += test['hit_length']
#     test_hit = DeltaCompressedHit(test)
#     test_dt = utctime_to_datetime_ns(test_hit.utc_time)
#     test_dc = compute_dom_launch_time(test_hit)
#     if test_hit.dom_id_str == '13c51a83e844':
#         # print(test_hit.dom_id_str)
#         # print('{0}   {1}'.format(test_dt, test_hit.utc_time))
#         print('{0}   {1}'.format(test_dc, test_hit.dom_clk*25))
#         # print((test_hit.pedestal, test_hit.version))