import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
from sklearn.ensemble import IsolationForest
from settings.settings import BASE_DIR


color_scheme = {0: '#7CB342', 1: '#FFA000', 2: '#7B1FA2'}

class ImgText:
    def __init__(self, **kwargs):
        self.img = ""
        self.text = ""
        self.title = ""
        self.table_title = ""
        self.table = ""

        if 'title' in kwargs.keys():
            self.title = kwargs['title']
        if 'img' in kwargs.keys():
            self.img = kwargs['img']
        if 'text' in kwargs.keys():
            self.text = kwargs['text']
        if 'table' in kwargs.keys():
            self.table_title = kwargs['table_title']
            self.table = kwargs['table']

# !TODO draw head of data here

# total package distr ----------------------------------------------------------




def get_global_path(name):
    return BASE_DIR + '/static/' + name


def unique_pac_per_time(df, glob_p, name, save_to_file=False, figsize=(20, 14)):
    test = np.array(list(map(int, df.Time)))
    n_bins = len(test) // 60
    if n_bins > 10**6:
        n_bins /= 10
    plt.figure(figsize=figsize)
    plt.hist(test, bins=n_bins, color=color_scheme[0])
    plt.grid(ls=':')
    if not save_to_file:
        plt.show()
    else:
        plt.savefig(glob_p + '/total_package_time.png')
        return 'Распределение количества переданных и полученных пакетов по времени', name + '/total_package_time.png'
        

def draw_package_size(data, glob_p, name, save_to_file=False):
    plt.figure(figsize=(15, 8))
    plt.hist(data['Length'], bins=30, color=color_scheme[1])
    plt.title('Len of the package distribution')
    plt.ylabel('count')
    plt.xlabel('len package')
    if not save_to_file:
        plt.show()
    else:
        plt.savefig(glob_p + '/total_package_size.png')
        return 'Распределение переданных и полученных пакетов по размерам', name + '/total_package_size.png'
        

def total_hist_by_protocol(data, glob_p, name, save_to_file=False):
    grouped_by_protocol = data.groupby(['Protocol'])['No.'].count().sort_values()
    plt.figure(figsize=(10, 5))
    
    objects = grouped_by_protocol.keys()
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, grouped_by_protocol.values, align='center', color=color_scheme[2])
    plt.xticks(y_pos, objects, rotation=90)

    plt.ylabel('Requests')
    plt.title('Hist by protocol')
    if not save_to_file:
        plt.show()
    else:
        plt.savefig(glob_p + '/total_by_protocol.png')
        return 'Распределение количества пакетов по протоколам', name + '/total_by_protocol.png'


def hist_by_source(data, glob_p, name, save_to_file=False):
    grouped_by_source = data.groupby(['Source'])['No.'].count()

    plt.figure(figsize=(15, 5))
    
    objects = grouped_by_source.keys()[grouped_by_source.values > 1000]
    y_pos = np.arange(len(objects))

    print(grouped_by_source.values[grouped_by_source.values > 1000])
    
    plt.bar(y_pos, grouped_by_source.values[grouped_by_source.values > 1000], align='center', color=color_scheme[0])
    plt.xticks(y_pos, objects, rotation=90)

    plt.ylabel('Requests')
    plt.title('Top')
    if not save_to_file:
        plt.show()
    else:
        plt.savefig(glob_p + '/total_by_source.png')
        return 'Наиболее часто встречающиеся source-IP', name + '/total_by_source.png'


def hist_by_dest(data, glob_p, name, save_to_file=False):
    grouped_by_dest =data.groupby(['Destination'])['No.'].count()

    plt.figure(figsize=(15, 5))
    
    objects = grouped_by_dest.keys()[grouped_by_dest.values > 1000]
    y_pos = np.arange(len(objects))

    print(grouped_by_dest.values[grouped_by_dest.values > 1000])
    
    plt.bar(y_pos, grouped_by_dest.values[grouped_by_dest.values > 1000], align='center', color=color_scheme[1])
    plt.xticks(y_pos, objects, rotation=90)

    plt.ylabel('Requests')
    plt.title('Top')
    if not save_to_file:
        plt.show()
    else:
        plt.savefig(glob_p + '/total_by_dest.png')
        return 'Наиболее часто встречающиеся destination-IP', name + '/total_by_dest.png'


def draw_by_protocol(data, protocol_type, glob_p, name, save_to_file=False,
                     bins=300, draw_from=None, draw_to=None, figsize=(15,5)):

    if draw_from == None:
        draw_from = data['Time'].min() - 1
    if draw_to == None:
        draw_to = data['Time'].max() + 1
    
    plt.figure(figsize=figsize)
    plt.title('Protocol type - ' + protocol_type)
    data_slice = data[np.logical_and(data['Protocol'] == protocol_type, 
            np.logical_and(draw_from < data['Time'], data['Time'] < draw_to))]
    plt.hist(data_slice['Time'], bins=bins, color=color_scheme[0])
    if not save_to_file:
        plt.show()
    else:
        filename = '/by_protocol_' + protocol_type + '.png'
        plt.savefig(glob_p + filename)
        return 'Распределение количества пакетов, переданных и полученных по протоколу ' + protocol_type\
            + ' по времени', name + filename
        

def plot_timeline(data, glob_p, name, save_to_file=False):
    fig, ax=plt.subplots(figsize=(15,6))

    labels=[]
    for i, task in enumerate(data.groupby(['Protocol'])):
        labels.append(task[0])
        r=task[1]
        dta = r[['Time']]
        dta['diff'] = np.ones_like(r['Time'])
        ax.broken_barh(dta.values, (i-0.4,0.8), color=color_scheme[2], alpha=0.1)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels) 
    ax.set_xlabel("time [s]")
    plt.tight_layout()
    if not save_to_file:
        plt.show()
    else:
        plt.savefig(glob_p + '/global_timeline.png')
        return 'Gantt chart количества пакетов для каждого протокола', name + '/global_timeline.png'



# smtp suspicious windows ------------------------------------------------------

# segs_smtp_susp = []


def smtp_get_susp(data, glob_p, name, save_to_file=False):
    SMTP_WINDOW_TIMEOUT_S = 15
    SMTP_WINDOW_MIN_SUSP_LEN_S = 1
    SMTP_WINDOW_MIN_SUSP_REQS = 30
    SMTP_WINDOW_MIN_SUSP_RATIO = 0.3
    segs_smtp_susp = []
    smtp_data = data[data['Protocol'] == 'SMTP'].drop(['Protocol', 'No.'], axis=1)
    smtp_times = smtp_data['Time']
    smtp_deltas = smtp_times.values[1:] - smtp_times.values[:-1]
    temp1 = [0] + list(np.argwhere(smtp_deltas > SMTP_WINDOW_TIMEOUT_S).ravel())
    smtp_segments = list(zip(temp1[:-1], temp1[1:]))

    idx = 0
    plt.figure(figsize=(12, 8))
    for start, stop in smtp_segments:
        idx += 1
        seg_range = np.arange(start, stop + 1)
        # print('Window #{}:'.format(idx))
        seg_time_len = smtp_data.iloc[stop]['Time'] - smtp_data.iloc[start]['Time']
        # print('    time len     {:.2f} s.'.format(seg_time_len))
        seg_reqs_ratio = (stop - start) / seg_time_len
        # print('    reqs ratio   {:.2f} / s. ({} total)'.format(seg_reqs_ratio, stop-start))
        email_addresses = np.count_nonzero(smtp_data.iloc[seg_range]['Info'].apply(lambda x: x.find(r'@') != -1))
        # print('    email refers {}'.format(email_addresses))
        if seg_time_len > SMTP_WINDOW_MIN_SUSP_LEN_S and stop - start > SMTP_WINDOW_MIN_SUSP_REQS \
                and seg_reqs_ratio > SMTP_WINDOW_MIN_SUSP_RATIO:
            segs_smtp_susp.append([smtp_data.iloc[start]['Time'], smtp_data.iloc[stop]['Time']])
            plt.hlines(seg_reqs_ratio, smtp_data.iloc[start]['Time'],
                       smtp_data.iloc[stop]['Time'], color='red', lw=4)

    if not save_to_file:
        plt.show()
    else:
        plt.savefig(glob_p + '/smtp_problem_windows.png')

    if (len(segs_smtp_susp) > 0):
        segs_smtp_susp = pd.DataFrame(segs_smtp_susp)
        segs_smtp_susp.columns = ['Время начала', 'Время конца']
        segs_smtp_susp['Причина'] = ['Spam'] * segs_smtp_susp.shape[0]
        return 'Предполагаемые моменты времени, когда происходит рассылка спама',\
               name + '/smtp_problem_windows.png', segs_smtp_susp
    else:
        return 'Предполагаемые моменты времени, когда происходит рассылка спама',\
                       name + '/smtp_problem_windows.png', pd.DataFrame({'Время начала': [], 'Время конца': [],
                                                                         'Причина': []})


def get_udp_top_loads(data, save_to_file=False):
    UDP_THRESHOLD_SINGLE_CONNECTION = 100
    UDP_SAMPLES_TO_DRAW = 30
    udp_data = data[data['Protocol'] == 'UDP']
    your_ip = data['Source'].value_counts().idxmax()
    udp_data_out = udp_data[udp_data['Source'] == your_ip]
    udp_data_out['Sport'] = udp_data_out['Info'].apply(lambda x: int(x.split(' ')[0]))
    udp_data_out['Dport'] = udp_data_out['Info'].apply(lambda x: int(x.split(' ')[4]))
    udp_data_reqs_dict = dict()
    for x in udp_data_out[['Time', 'Destination', 'Sport', 'Dport']].groupby(['Destination', 'Dport']):
        if (x[1].shape[0] > UDP_THRESHOLD_SINGLE_CONNECTION):
            udp_data_reqs_dict[x[0]] = x[1].shape[0]
    udp_dct_ks = list(udp_data_reqs_dict.keys())
    if len(udp_data_reqs_dict) > 0:
        udp_smpl, _ = zip(*list(sorted(udp_data_reqs_dict.items(), key=operator.itemgetter(1)))[-UDP_SAMPLES_TO_DRAW:])

        plt.figure(figsize=(12, 8))
        i = 0
        for d_ip, d_port in udp_smpl:
            i += 1
            todraw_data = udp_data_out[udp_data_out['Destination'] == d_ip]['Time']
            plt.scatter(todraw_data, i*np.ones_like(todraw_data), marker='o', alpha=0.2, color=color_scheme[1])
        plt.title('Most popular UDP dest IP/port pair requests distrib')
        if not save_to_file:
            plt.show()
        else:
            plt.savefig('./udp_sample_windows.png')
        return True
    return False

# if get_udp_top_loads(data, True) ...

# ISO --------------------------------------------------------------------------




def apply_ISOForest(df, contamination=0.1, columns_to_use=[]):
    '''
    input -- pandas data frame with columns [No. Time Source Destination Protocol Length Info]

    output -- numpy array with {0, 1} where 1 is anomaly
    '''
    iso_frame = df.copy()

    iso_frame.drop(columns=['No.', 'Source', 'Destination', 'Info'], inplace=True)
    iso_frame = pd.get_dummies(iso_frame, columns=['Protocol'])

    iso_frame['Time'] = list(map(int, iso_frame['Time'].values))

    names = []
    if type(columns_to_use) == str:
        names = ['Length', 'Protocol_' + columns_to_use]
    if type(columns_to_use) == list:
        if len(columns_to_use) == 0:
            names = list(iso_frame.columns)
            names.remove('Time')
        else:
            names2 = []
            for name in names:
                if len(name.split('_')) != 0:
                    continue
                elif name.split('_')[1] in columns_to_use:
                    name2.append('Protocol_' + name)
            names = names2
            names.append('Length')

    iso_frame2 = pd.DataFrame(columns=names)

    grouped = iso_frame.groupby('Time')

    for col in names:
        iso_frame2[col] = grouped[col].agg(np.sum)

    iso = IsolationForest(n_jobs=-1, n_estimators=20)
    iso.fit(iso_frame2)
    pred_out = iso.predict(iso_frame2)
    pred_out[pred_out == 1] = 0
    pred_out[pred_out == -1] = 1
    return pred_out


def create_segments(ar, gap_ignore=0.03, remove_less_when=0.02):
    '''
    input -- array of {0, 1} where 1 is anomaly

    output -- numpy array [[s1, f1], ..., [sn, fn]] where [s1, f1] - segment of anomaly
    '''
    ar = list(ar)
    ar.append(0)
    ar = np.array(ar)
    whole_len = len(ar)
    segm1 = []
    start = 0
    now_seg = False
    for i, x in enumerate(ar):
        if x == 1:
            if now_seg == False:
                start = i
                now_seg = True
                continue
        if x == 0:
            if now_seg == True:
                segm1.append([start, i - 1])
                now_seg = False
                continue
    segm2 = []
    if len(segm1) == 0:
        return []
    start = segm1[0][0]
    now_seg = False
    for i in range(len(segm1) - 1):
        if abs(segm1[i][1] - segm1[i + 1][0]) <= whole_len * gap_ignore:
            if now_seg == True:
                continue
            else:
                now_seg = True
                start = segm1[i][0]
                continue
        else:
            if now_seg == False:
                segm2.append(segm1[i])
                continue
            else:
                now_seg = False
                segm2.append([start, segm1[i][1]])
                start = -1
                continue
    if now_seg == True:
        segm2.append([start, segm1[-1][1]])
    if segm2[-1][1] != segm1[-1][1]:
        segm2.append(segm1[-1])
    segm3 = []
    for seg in segm2:
        if seg[1] - seg[0] >= remove_less_when * whole_len:
            segm3.append(seg)
    return segm3



def get_summary(df, glob_p, name, save_to_file=False, figsize=(16, 10)):
    fig, ax = plt.subplots(figsize=figsize)

    labels=[]
    index_of_total = len(np.unique(df.Protocol))
    for i, proto in enumerate(np.unique(df.Protocol)):
        segms = create_segments(apply_ISOForest(df,
                                                columns_to_use=proto,
                                                contamination=0.015),
                                gap_ignore=0.003,
                                remove_less_when=0.01)
        labels.append(proto)
        for i in range(len(segms)):
            segms[i][1] -= segms[i][0]
            segms[i] = tuple(segms[i])
        ax.broken_barh(segms, (i-0.4,0.8), color='blue', alpha=0.1)
        ax.broken_barh(segms, (index_of_total-0.4,0.8), color='red', alpha=0.07)

    labels.append('Total')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("time [s]", fontsize=16)
    plt.tight_layout()
    if not save_to_file:
        plt.show()
    else:
        plt.savefig(glob_p + '/gantt_phenomena.png')
        return 'Gantt chart аномалий для каждого протокола', name + '/gantt_phenomena.png'


def dataframe_to_list(data):
    # print(data)
    df = []
    for ind in data.index:
        df.append([])
        for col in data.columns:
            df[-1].append(data.iloc[ind][col])
    return list(data.columns), df


def get_summary_DF(df):
    whole_len = int(np.array(df.Time).max())
    final_seg = np.zeros(whole_len)
    reason = []
    for i, proto in enumerate(np.unique(df.Protocol)):
        segms = create_segments(apply_ISOForest(df,
                                                columns_to_use=proto,
                                                contamination=0.01),
                                gap_ignore=0.004,
                                remove_less_when=0.01)
        sum_len = 0
        for s in segms:
            final_seg[s[0]:s[1]] += 1
            sum_len += s[1] - s[0]
        reason.append((sum_len, proto))
    # print(reason)
    pixels = final_seg >= final_seg.mean() + 2 * final_seg.std()
    final_seg[pixels] = 1
    final_seg[np.logical_not(pixels)] = 0
    final_seg = create_segments(ar=final_seg, gap_ignore=0.0005, remove_less_when=0.001)
    ans_df = pd.DataFrame(columns=['Время начала', 'Время конца', 'Причина'])
    for seg in final_seg:
        ans_df = ans_df.append({'Время начала' : seg[0],
                       'Время конца' : seg[1],
                       'Причина' : 'Повышенное количество аномалий в протоколах : ' +
                       ' '.join([x[1] for x in reason if x[0] > whole_len * 0.009 ])
                               },
                              ignore_index=True)
    return ans_df



def call_all_funcs(data, name):
    glob_p = get_global_path(name)
    images = []

    your_ip = data['Source'].value_counts().idxmax()
    images.append(ImgText(**{'title': 'Наш IP', 'text': your_ip}))

    title, img = unique_pac_per_time(data, glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = draw_package_size(data, glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = total_hist_by_protocol(data, glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = hist_by_source(data, glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = hist_by_dest(data, glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = draw_by_protocol(data, 'TCP', glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = draw_by_protocol(data, 'UDP', glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = draw_by_protocol(data, 'SMTP', glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = draw_by_protocol(data, 'HTTP', glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = draw_by_protocol(data, 'DNS', glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = draw_by_protocol(data, 'ICMP', glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = draw_by_protocol(data, 'DHCPv6', glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = plot_timeline(data, glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))

    title, img = get_summary(data, glob_p, name, True)
    images.append(ImgText(**{'title': title, 'img': img}))


    dataframe_m = get_summary_DF(data)
    title, img, dataframe_ed = smtp_get_susp(data, glob_p, name, True)
    dataframe_m = dataframe_m.append(dataframe_ed)
    table_title, table = dataframe_to_list(dataframe_m)

    images.append(ImgText(**{'title': title, 'img': img, 'table_title': table_title, 'table': table}))

    return images
