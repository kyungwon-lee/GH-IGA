import pandas as pd
from itertools import chain, tee, combinations, groupby
from functools import reduce, wraps
from operator import mul
from collections import defaultdict, Counter
import random
import time
import numpy as np
import copy
import json
import os
import sys

def show_time(func):

    """
    timer decorator
    """

    # TODO: 구조 알아보기. 함수 여기서 실행하는 건가?
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        함수 앞에 붙여주면 함수를 실제로 실행하고(??)
        <running 함수이름.. elapsed time: 0.몇몇 sec> 이런 식으로 출력해주는 함수
        """

        t0 = time.time()

        print(f'running {func.__name__}.. ', end='')
        # print('runnning {func.__name__}..')
        sys.stdout.flush()

        v = func(*args, **kwargs) # 함수를 실행??

        m, s = divmod(time.time() - t0, 60)

        st = 'elapsed time:'

        if m:
            st += ' ' + f'{m:.0f} min'
            # st += " {m:.0f} min"
        if s:
            st += ' ' + f'{s:.3f} sec'
            # st += " {s:.3f} sec"

        print(st)

        return v

    return wrapper

# 아래 클래스를 __init__하고 원하는 모델에 해당하는 method 실행하여 적합. 'show' 메소드로 결과 보기
class MTA:

    def __init__(self, data, sep=' > ', save_groupby=False):

        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data', data))
        self.sep = sep
        self.NULL = '(null)'
        self.START = '(start)'
        self.CONV = '(conversion)'
        

        # 이미 정해진 칼럼 명에 해당되지 않는 칼럼이 데이터에 있으면 raise value error
        # set에서 <= 는 왼쪽이 오른쪽의 부분집합이라는 뜻
        if not (set(self.data.columns) <= set('path total_conversions total_conversion_value total_null'.split())):
            # raise ValueError(f'wrong column names in {data}!')
            raise ValueError('wrong column names in {data}!')

        # 만약 꼭 필요한 칼럼이 없는 경우 이미 있는 데이터로 대략적으로 채워넣기
        lack = set('path total_conversions total_conversion_value total_null'.split()) - set(self.data.columns)
        for column in lack:
            exec("self.add_{}()".format(column)) # add_total_conversions_value() 혹은 add_total_null() 호출

    
        # 무조건 remove loops
        self.remove_loops()

        # we'll work with lists in path from now on
        # path를 단일 string에서 channel 리스트로 쪼개기
        self.data['path'] = self.data['path'].apply(lambda _: [ch.strip() for ch in _.split(self.sep.strip())])

        if save_groupby:
            filepath = os.path.join(os.path.dirname(__file__),'res', data.split('.')[0] + '_groupby.csv')
            self.data.to_csv(filepath, index=False)

        # make a sorted list of channel names
        self.channels = sorted(list({ch for ch in chain.from_iterable(self.data['path'])}))
        
        # add some extra channels
        self.channels_ext = [self.START] + self.channels + [self.CONV, self.NULL]
        # make dictionary mapping a channel name to it's index
        self.c2i = {c: i for i, c in enumerate(self.channels_ext)}
        # and reverse
        self.i2c = {i: c for c, i in self.c2i.items()}

        self.removal_effects = defaultdict(float)

        self.attribution = defaultdict(lambda: defaultdict(float))

######################## init helper functions ######################## 

    def show(self):

        """
        show simulation results
        """

        res = pd.DataFrame.from_dict(mta.attribution)

        print(res)

    def save(self, filename):

        """
        save simulation results as csv
        """

        res = pd.DataFrame.from_dict(mta.attribution)
        filepath = os.path.join(os.path.dirname(__file__),'res', filename +'.csv')
        
        res.reset_index(level=0, inplace=True)
        res.rename(columns={'index':'channel'}, inplace=True)
        res.to_csv(filepath, index=False)
        

    def __repr__(self):

        # return f'{self.__class__.__name__} with {len(self.channels)} channels: {", ".join(self.channels)}'
        return '{self.__class__.__name__} with {len(self.channels)} channels: {", ".join(self.channels)}'

    def add_total_conversion_value(self):
        """
        total_conversion_value 칼럼이 필요하다면 conversion 값과 동일하게 채우기
        """
        self.data['total_conversion_value'] = self.data['total_conversions']

    def add_total_null(self):
        """
        total_null 칼럼이 필요하다면 conversion이 0인 경우 1, 0이 아닌 경우 0으로 채우기
        """
        self.data['total_null'] = self.data['total_conversions'].apply(lambda x: int(not bool(x)))


    @show_time
    def remove_loops(self):

        """
        remove transitions from a channel directly to itself, e.g. a > a
        """

        # list로 쪼개기
        self.data['path'] = self.data['path'].apply(lambda _: [ch.strip() for ch in _.split(self.sep.strip())])
        
        # groupby로 중복 제거한 후 다시 string으로 모으기
        self.data['path'] = self.data['path'].apply(lambda path: self.sep.join([x[0] for x in groupby(path)]))

        # path 별로 conversions과 null 집계
        self.data = self.data.groupby('path').sum().reset_index()
        return self


######################## heuristic ######################## 

    @show_time
    def linear(self, share='same', normalize=True):

        """
        either give exactly the same share of conversions to each visited channel (option share=same) or
        distribute the shares proportionally, i.e. if a channel 1 appears 2 times on the path and channel 2 once
        then channel 1 will receive double credit

        note: to obtain the same result as ChannelAttbribution produces for the test data set, you need to

            - select share=proportional
            - allow loops - use the data set as is without any modifications
        """

        if share not in 'same proportional'.split():
            raise ValueError('share parameter must be either *same* or *proportional*!')

        self.linear = defaultdict(float)

        for row in self.data.itertuples():

            if row.total_conversions:

                if share == 'same':

                    n = len(set(row.path))    # number of unique channels visited during the journey
                    s = row.total_conversions/n    # each channel is getting an equal share of conversions

                    for c in set(row.path):
                        self.linear[c] += s

                elif share == 'proportional':

                    c_counts = Counter(row.path)  # count how many times channels appear on this path
                    tot_appearances = sum(c_counts.values())

                    c_shares = defaultdict(float)

                    for c in c_counts:

                        c_shares[c] = c_counts[c]/tot_appearances

                    for c in set(row.path):

                        self.linear[c] += row.total_conversions*c_shares[c]

        if normalize:
            self.linear = self.normalize_dict(self.linear)

        self.attribution['linear'] = self.linear

        return self

    @show_time
    def position_based(self, r=(40,40), normalize=True):

        """
        give 40% credit to the first and last channels and divide the rest equally across the remaining channels
        """

        self.position_based = defaultdict(float)

        for row in self.data.itertuples():

            if row.total_conversions:

                n = len(set(row.path))

                if n == 1:
                    self.position_based[row.path[-1]] += row.total_conversions
                elif n == 2:
                    equal_share  = row.total_conversions/n
                    self.position_based[row.path[0]] += equal_share
                    self.position_based[row.path[-1]] += equal_share
                else:
                    self.position_based[row.path[0]] += r[0]*row.total_conversions/100
                    self.position_based[row.path[-1]] += r[1]*row.total_conversions/100

                    for c in row.path[1:-1]:
                        self.position_based[c] += (100 - sum(r))*row.total_conversions/(n - 2)/100

        if normalize:
            self.position_based = self.normalize_dict(self.position_based)

        self.attribution['pos_based'] = self.position_based

        return self


    @show_time
    def time_decay(self, count_direction='left', normalize=True):

        """
        time decay - the closer to conversion was exposure to a channel, the more credit this channel gets

        this can work differently depending how you get timing sorted. 

        example: a > b > c > b > a > c > (conversion)
    
        we can count timing backwards: c the latest, then a, then b (lowest credit) and done. Or we could count left to right, i.e.
        a first (lowest credit), then b, then c. 

        """

        self.time_decay = defaultdict(float)

        if count_direction not in 'left right'.split():
            raise ValueError('argument count_direction must be *left* or *right*!')

        for row in self.data.itertuples():

            if row.total_conversions:

                channels_by_exp_time = []

                _ = row.path if count_direction == 'left' else row.path[::-1]

                for c in _:
                    if c not in channels_by_exp_time:
                        channels_by_exp_time.append(c)

                if count_direction == 'right':
                    channels_by_exp_time = channels_by_exp_time[::-1]

                # first channel gets 1, second 2, etc.

                score_unit = 1./sum(range(1, len(channels_by_exp_time) + 1))

                for i, c in enumerate(channels_by_exp_time, 1):
                    self.time_decay[c] += i*score_unit*row.total_conversions

        if normalize:
            self.time_decay = self.normalize_dict(self.time_decay)

        self.attribution['time_decay'] = self.time_decay

        return self

    @show_time
    def first_touch(self, normalize=True):

        first_touch = defaultdict(int)

        for c in self.channels:

            # total conversions for all paths where the first channel was c
            first_touch[c] = self.data.loc[self.data['path'].apply(lambda _: _[0] == c), 'total_conversions'].sum()

        if normalize:
            first_touch = self.normalize_dict(first_touch)

        self.attribution['first_touch'] = first_touch

        return self

    @show_time
    def last_touch(self, normalize=True):

        last_touch = defaultdict(int)

        for c in self.channels:

            # total conversions for all paths where the last channel was c
            last_touch[c] = self.data.loc[self.data['path'].apply(lambda _: _[-1] == c), 'total_conversions'].sum()

        if normalize:
            last_touch = self.normalize_dict(last_touch)

        self.attribution['last_touch'] = last_touch

        return self

######################## shapely helper functions ######################## 
    def get_generated_conversions(self, max_subset_size=3):

        # 그냥 깡통같은 거 생성
        self.cc = defaultdict(lambda: defaultdict(float))

        # path 하나 씩 돌며 (채널들의 목록, 전환이 발생한 레코드수, 발생하지 않은 레코드수)를 같이 가져오기
        for ch_list, convs, nulls in zip(self.data['path'], 
                                            self.data['total_conversions'], 
                                                self.data['total_null']):

            # only look at journeys with conversions
            for n in range(1, max_subset_size + 1): # subset의 크기를 1부터 max까지 하나씩 늘려가며

                for tup in combinations(set(ch_list), n): # path로부터 크기가 n인 subset을 하나씩 돌기

                    tup_ = self.ordered_tuple(tup) # ordered_tuple에 저장

                    # cc의 key는 주어진 path로 부터 가능한 subset
                    # 그에 매칭되는 cc의 value는 다시 딕셔너리인데 {'(conv)': conversion 발생수, '(null)': null로 빠진 수} 형태
                    # TODO: conversions 수 말고 value로 바꾸기
                    self.cc[tup_][self.CONV] += convs # 해당 subset의 conversion을 c 저장
                    
                    
                    self.cc[tup_][self.NULL] += nulls # 해당 subset의 null 저장

        return self

    
    def v(self, coalition):
        
        """
        total number of conversions generated by all subsets of the coalition;
        coalition is a tuple of channels
        """

        s = len(coalition)

        total_convs = 0

        for n in range(1, s+1):
            for tup in combinations(coalition, n):
                tup_ = self.ordered_tuple(tup)
                # TODO: conversions 수 말고 value로 바꾸기
                total_convs += self.cc[tup_][self.CONV]

        return total_convs

    def w(self, s, n):
        # TODO: 왜 이걸 곱해주는가?
        return np.math.factorial(s)*(np.math.factorial(n - s -1))/np.math.factorial(n)



######################## shapely ######################## 
    @show_time
    def shapley(self, max_coalition_size=2, normalize=True):

        """
        max_coalition_size는 결국 타협의 문제
        - path 길이 최대치보다는 하나 작아야한다
        - path가 어차피 길지 않기도 하고, 조합이 길어지면 등장 빈도가 낮아지니까 고려를 덜 해도 된다
        Shapley model; channels are players, the characteristic function maps a coalition A to the 
        the total number of conversions generated by all the subsets of the coalition

        see https://medium.com/data-from-the-trenches/marketing-attribution-e7fa7ae9e919
        """

        # 결과는 self.cc라는 dictionary에 저장되어 있다
        # key: 가능한 모든 subset
        # value: 다시 dictionary key: (conversion) value:그게 포함된 path에서 발생된 conversion의 합, key: (null) value: null의 합
        # NOTE: max_subset은 max_coalition_size보다 적어도 하나 커야함(v에서 최대 max_coalition_size+1 크기의 튜플을 cc에 key로 전달)
        self.get_generated_conversions(max_subset_size=max_coalition_size+1) 

        # 최종적으로 반환하는 매체를 key로 하고 기여도를 value로 하는 dictionary 깡통
        self.phi = defaultdict(float)

        for ch in self.channels:
            # subset의 size를 1부터 하나씩 키워가며
            for n in range(1, max_coalition_size + 1): # 
                # 현재 보고 있는 ch을 제외한 것으로 만들 수 있는, size가 n인 combination(튜플형태)을 하나 씩 돌면서 
                for tup in combinations(set(self.channels) - {ch}, n):
                    # 해당 (combination+현재채널)이 발생시킬 수 있는 잠재적 conversion 능력(self.v)와 ch 없이 해당 combination이 발생시킬 수 있는 잠재적 conversion 능력의 차이
                    # 에다가 해당 튜플의 길이(n)와 전체 채널 개수로 도출한 w를 weight처럼 곱해주기
                    self.phi[ch] += (self.v(tup + (ch,)) - self.v(tup))*self.w(len(tup), len(self.channels))

        if normalize:
            self.phi = self.normalize_dict(self.phi)

        self.attribution['shapley'] = self.phi

        return self

######################## markov helper functions ######################## 

    def normalize_dict(self, d):
        """
        returns a value-normalized version of dictionary d
        """
        sum_all_values = sum(d.values())

        for _ in d:
            d[_] = round(d[_]/sum_all_values, 6)

        return d

    def pairs(self, lst):

        it1, it2 = tee(lst) # 하나의 리스트로 두 개의 iterator 만들기(개수 default가 2임)
        next(it2, None) # 두 번째 iterator는 한 스텝 수행

        return zip(it1, it2)

    def count_pairs(self):

        """
        count how many times channel pairs appear on all recorded customer journey paths
        """

        c = defaultdict(int)

        for row in self.data.itertuples(): # 관측치를 하나 씩 돌며

            # start + path에서 생성할수 있는 pair당 발생 수 집계
            for ch_pair in self.pairs([self.START] + row.path): # start에 현재의 path를 붙여서 연속된 두 씩 pair 생성 
                c[ch_pair] += (row.total_conversions + row.total_null)
            
            # path 끝단 처리: 마지막 포인트에서 '(null)'으로의 횟수와 '(conv)'으로의 발생수 집계
            c[(row.path[-1], self.NULL)] += row.total_null
            c[(row.path[-1], self.CONV)] += row.total_conversions

        # key: '(start)', '(conv)', '(null)'을 포함하여 path에서 생성할 수 있는 페어
        # value: 각 페어의 발생횟수
        return c 

    def ordered_tuple(self, t):

        """
        return tuple t ordered 
        """

        sort = lambda t: tuple(sorted(list(t)))

        return (t[0],) + sort(t[1:]) if (t[0] == self.START) and (len(t) > 1) else sort(t)


    def trans_matrix(self):

        """
        calculate transition matrix which will actually be a dictionary mapping 
        a pair (a, b) to the probability of moving from a to b, e.g. T[(a, b)] = 0.5
        """
        # 최종적으로 반환하는 결과값: pair to probability mapping
        tr = defaultdict(float) 

        # 분모로 쓰이는 딕셔너리: 터치포인트 - 첫번째로 등장하는 경우의 수 mapping
        outs = defaultdict(int) 

        # here pairs are unordered
        pair_counts = self.count_pairs()

        # pass 1: outs 세기
        for pair in pair_counts:

            outs[pair[0]] += pair_counts[pair]

        # pass 2: a->b의 probability = pair (a,b)의 등장횟수 / 첫번째 아이템으로 a의 총 등장횟수
        for pair in pair_counts:

            tr[pair] = pair_counts[pair]/outs[pair[0]]

        return tr

    @show_time
    def simulate_path(self, trans_mat, drop_channel=None, n=int(1e6)):

        """
        generate n random user journeys and see where these users end up - converted or not;
        drop_channel is a channel to exclude from journeys if specified
        """

        outcome_counts = defaultdict(int)

        idx0 = self.c2i[self.START]
        null_idx = self.c2i[self.NULL]
        conv_idx = self.c2i[self.CONV]

        drop_idx = self.c2i[drop_channel] if drop_channel else null_idx

        for _ in range(n):

            stop_flag = None

            while not stop_flag:

                probs = [trans_mat.get((self.i2c[idx0], self.i2c[i]), 0) for i in range(len(self.channels_ext))]

                # index of the channel where user goes next
                idx1 = np.random.choice([self.c2i[c] for c in self.channels_ext], p=probs, replace=False)

                if idx1 == conv_idx:
                    outcome_counts[self.CONV] += 1
                    stop_flag = True
                elif idx1 in {null_idx, drop_idx}:
                    outcome_counts[self.NULL] += 1
                    stop_flag = True
                else:
                    idx0 = idx1

        return outcome_counts

    def prob_convert(self, trans_mat, drop=None):

        # total_conversions가 0보다 크고 drop 터치포인트에 해당하지 않는 data만 추리기
        # conversions는 없고 null만 있는 행도 data에 기록되어 있기 때문에
        # drop이 None이라면 첫번째 조건은 있으나마나
        _d = self.data[self.data['path'].apply(lambda x: drop not in x) & (self.data['total_conversions'] > 0)]

        p = 0

        for row in _d.itertuples():

            pr_this_path = [] # 하나의 path를 pair씩 분해했을때, 해당 pair가 발생할 확률을 순차적으로 저장할 리스트

            # path의 맨앞과 맨뒤에 '(start)', '(conv)' 아이템을 붙인 리스트로 pair 생성
            for t in self.pairs([self.START] + row.path + [self.CONV]):
                
                # 생성한 trans_mat 딕셔너리에서 해당 pair의 확률 찾기. 
                # dictionary에 해당 pair가 없으면 확률 0이라는 뜻
                pr_this_path.append(trans_mat.get(t, 0))

            # 한 리스트의 pair 당 확률을 모두 곱한 값(==해당 path가 발생할 확률)의 총합
            p += reduce(mul, pr_this_path)

        return p

######################## markov ######################## 
    
    #@show_time
    def markov(self, normalize=True):
        
        # 채널 별 기여도 계산 결과를 저장할 dictionary 변수
        markov = defaultdict(float)

        # calculate the transition matrix
        tr = self.trans_matrix()

        # 모든 터치포인트가 들어갔을 때 전환이 발생할 path를 따를 확률
        p_conv = self.prob_convert(trans_mat=tr) 

        for c in self.channels:
            # drop = c로 해당 터치포인트를 제거했을 때 전환이 발생할 path를 따를 확률을 구하고
            # 그 감소폭의 비율이 곧 터치포인트의 기여도이다
            markov[c] = (p_conv - self.prob_convert(trans_mat=tr, drop=c))/p_conv
        
        if normalize:
            # normalize_dict: dictionary의 모든 value의 합으로 각각을 나누는 함수
            markov = self.normalize_dict(markov) 

        # 클래스 변수인 attribution에 결과값 저장
        self.attribution['markov'] = markov


        return self



if __name__ == '__main__':
    import sys

    # default: sep=' > ', allow_loops=False
    mta = MTA(data='LB_201908.csv', sep = '>', save_groupby=True)
    # mta = MTA(data='LB_201908.csv', sep = '>', save_groupby=True)

    mta.linear(share='proportional') \
            .time_decay(count_direction='right') \
            .first_touch() \
            .position_based() \
            .last_touch() \
            .shapley() \
            .markov() \
            .show()

    # .csv 없이
    mta.save('result_LB_201908')

    