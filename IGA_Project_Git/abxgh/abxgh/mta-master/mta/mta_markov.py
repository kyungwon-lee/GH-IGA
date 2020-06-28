import pandas as pd
from itertools import chain, tee, combinations
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

import arrow # date & time 핸들링하는 패키지

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

		# print(f'running {func.__name__}.. ', end='')
		print('runnning {func.__name__}..')
		sys.stdout.flush()

		v = func(*args, **kwargs) # 함수를 실행??

		m, s = divmod(time.time() - t0, 60)

		st = 'elapsed time:'

		if m:
			# st += ' ' + f'{m:.0f} min'
			st += " {m:.0f} min"
		if s:
			# st += ' ' + f'{s:.3f} sec'
			st += " {s:.3f} sec"

		print(st)

		return v

	return wrapper

# 아래 클래스를 __init__하고 원하는 모델에 해당하는 method 실행하여 적합. 'show' 메소드로 결과 보기
class MTA:

	def __init__(self, data='data.csv.gz', allow_loops=False, add_timepoints=True, sep=' > '):

		self.data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data', data))
		self.sep = sep
		self.NULL = '(null)'
		self.START = '(start)'
		self.CONV = '(conversion)'

		# 이미 정해진 칼럼 명에 해당되지 않는 칼럼이 데이터에 있으면 raise value error
		# set에서 <= 는 왼쪽이 오른쪽의 부분집합이라는 뜻
		if not (set(self.data.columns) <= set('path total_conversions total_conversion_value total_null exposure_times'.split())):
			# raise ValueError(f'wrong column names in {data}!')
			raise ValueError('wrong column names in {data}!')

		if add_timepoints:
			self.add_exposure_times(1) # 하나의 path 당 하나의 가상 노출 시간을 부여. 1은 노출시간의 간격

		if not allow_loops:
			self.remove_loops() # loop 제거

		# we'll work with lists in path and exposure_times from now on
		# NOTE: 여기서 'exposure_times' 칼럼이 없으면 어케되는 거임?
		self.data[['path', 'exposure_times']] = self.data[['path', 'exposure_times']].applymap(lambda _: [ch.strip() for ch in _.split(self.sep.strip())])
		
		# make a sorted list of channel names
		self.channels = sorted(list({ch for ch in chain.from_iterable(self.data['path'])}))
		
		# add some extra channels
		self.channels_ext = [self.START] + self.channels + [self.CONV, self.NULL]
		# make dictionary mapping a channel name to it's index
		self.c2i = {c: i for i, c in enumerate(self.channels_ext)}
		# and reverse
		self.i2c = {i: c for c, i in self.c2i.items()}

		self.removal_effects = defaultdict(float)
		# touch points by channel
		self.tps_by_channel = {'c1': ['beta', 'iota', 'gamma'], 
								'c2': ['alpha', 'delta', 'kappa', 'mi'],
								'c3': ['epsilon', 'lambda', 'eta', 'theta', 'zeta']}

		self.attribution = defaultdict(lambda: defaultdict(float))

	def __repr__(self):

		# return f'{self.__class__.__name__} with {len(self.channels)} channels: {", ".join(self.channels)}'
		return '{self.__class__.__name__} with {len(self.channels)} channels: {", ".join(self.channels)}'

	def add_exposure_times(self, dt=None):

		"""
		generate synthetic exposure times; if dt is specified, the exposures will be dt=1 sec away from one another, otherwise
		we'll generate time spans randomly

		- the times are of the form 2018-11-26T03:54:26.532091+00:00
		"""

		if 'exposure_times' in self.data.columns:
			return self

		ts = []    # this will be a list of time instant lists one per path 

		if dt:

			_t0 = arrow.utcnow()

			self.data['path'].str.split('>') \
				.apply(lambda _: [ch.strip() for ch in _]) \
				.apply(lambda lst: ts.append(self.sep.join([r.format('YYYY-MM-DD HH:mm:ss') 
									for r in arrow.Arrow.range('second', _t0, _t0.shift(seconds=+(len(lst) - 1)))])))

		self.data['exposure_times'] = ts

		return self

	@show_time
	def remove_loops(self):

		"""
		remove transitions from a channel directly to itself, e.g. a > a
		"""

		cpath = []
		cexposure = []

		self.data[['path', 'exposure_times']] = self.data[['path', 'exposure_times']].applymap(lambda _: [ch.strip() for ch in _.split('>')]) 

		for row in self.data.itertuples():

			clean_path = []
			clean_exposure_times = []

			for i, p in enumerate(row.path, 1):

				if i == 1:
					clean_path.append(p)
					clean_exposure_times.append(row.exposure_times[i-1])
				else:
					if p != clean_path[-1]:
						clean_path.append(p)
						clean_exposure_times.append(row.exposure_times[i-1])

			cpath.append(self.sep.join(clean_path))
			cexposure.append(self.sep.join(clean_exposure_times))

		self.data_ = pd.concat([pd.DataFrame({'path': cpath}), 
								self.data[[c for c in self.data.columns if c not in 'path exposure_times'.split()]],
								pd.DataFrame({'exposure_times': cexposure})], axis=1)

		_ = self.data_[[c for c in self.data.columns if c != 'exposure_times']].groupby('path').sum().reset_index()

		self.data = _.join(self.data_[['path', 'exposure_times']].set_index('path'), 
											on='path', how='inner').drop_duplicates(['path'])

		return self

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

# 	def ordered_tuple(self, t):

# 		"""
# 		return tuple t ordered 
# 		"""

# 		sort = lambda t: tuple(sorted(list(t)))

# 		return (t[0],) + sort(t[1:]) if (t[0] == self.START) and (len(t) > 1) else sort(t)


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

	@show_time
	def markov(self, sim=False, normalize=True):
		
		# 채널 별 기여도 계산 결과를 저장할 dictionary 변수
		markov = defaultdict(float)

		# calculate the transition matrix
		# TODO: 시간 얼마걸리는지 재보기. 만약 데이터가 너무 많을 때 도출에 시간이 넘 오래걸리면 어쩌지?
		tr = self.trans_matrix()

		if not sim: # simulation 없이 deterministic한 방법
			
			# 모든 터치포인트가 들어갔을 때 전환이 발생할 path를 따를 확률
			p_conv = self.prob_convert(trans_mat=tr) 

			for c in self.channels:
				# drop = c로 해당 터치포인트를 제거했을 때 전환이 발생할 path를 따를 확률을 구하고
				# 그 감소폭의 비율이 곧 터치포인트의 기여도이다
				markov[c] = (p_conv - self.prob_convert(trans_mat=tr, drop=c))/p_conv
		else:

			outcomes = defaultdict(lambda: defaultdict(float))
			# get conversion counts when all chennels are in place
			outcomes['full'] = self.simulate_path(trans_mat=tr, drop_channel=None)

			for c in self.channels:

				outcomes[c] = self.simulate_path(trans_mat=tr, drop_channel=c)
				# removal effect for channel c
				markov[c] = (outcomes['full'][self.CONV] - outcomes[c][self.CONV])/outcomes['full'][self.CONV]

		if normalize:
			# normalize_dict: dictionary의 모든 value의 합으로 각각을 나누는 함수
			markov = self.normalize_dict(markov) 

		# 클래스 변수인 attribution에 결과값 저장
		self.attribution['markov'] = markov


		return self



# if __name__ == '__main__':

# 	# default: sep=' > ', add_timepoints=True
# 	mta = MTA(data='data.csv.gz', allow_loops=False)

# 	mta.linear(share='proportional') \
# 			.time_decay(count_direction='right') \
# 			.shapley() \
# 			.shao() \
# 			.first_touch() \
# 			.position_based() \
# 			.last_touch() \
# 			.markov(sim=False) \
# 			.show()

# 			# .logistic_regression() \ # 3분정도 걸림
# 			# .additive_hazard() \ # 오래 걸림
