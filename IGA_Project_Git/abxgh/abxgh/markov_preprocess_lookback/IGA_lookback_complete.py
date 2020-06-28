# 필요한 라이브러리를 가져오자.
# numpy, pandas
import numpy as np
import pandas as pd

# 이 함수는 dataframe의 datetime 열에 저장된 자료형을 string -> timestamp 로 바꾸어준다. 
import datetime
def make_datetime(row):
    return datetime.datetime.strptime(row, '%Y-%m-%d %H:%M:%S.%f')

#TODO: 나중에 Inactivity 함수와 함께 한 클래스로 통합하게 되면 데이터는 인스턴스 생성 시 argument로 주어지는 것으로 수정하자.

#raw_data_name: 불러올 데이터 파일의 이름. 형식이 정해진 것이 아니어서 확장자도 포함해 주어야 함.
#result_data_name: 결과를 저장할 파일의 이름. 형식이 정해진 것이 아니어서 확장자도 포함해 주어야 함.
#lookback_window: 우리가 지정할 lookback window의 크기. 며칠인지 일단위 정수.
def pathmaker_lookback(raw_data_name = 'data.csv', result_data_name = 'result.csv', lookback_window = 3, delete_abusing = False, abusing_threshold = 3):
    #이 때 data는 adid, datetime 순으로 정렬된 dataframe이라고 가정.
    data = pd.read_csv(raw_data_name)
    #기여한 곳 : 1, 기여하지 않은 곳 : 0 을 각각 표시해주기 위한 column인 contributed를 추가하고, 0으로 초기화
    data['contributed']=0

    #이제 datetime 열의 내용들은 string이 아니라 datetime object가 되어 연산이 빨라짐.
    data['datetime'] = data['datetime'].apply(make_datetime)

    
    #df를 뒤집는 코드
    data = data.reindex(index=data.index[::-1])
    data = data.reset_index(drop=True)

    #zip은 튜플로 구성된 iterator를 반환해준다.
    zipped_data = zip(range(len(data)),data['adid'],data['datetime'], data['partner_index'])

    standard_adid = ""
    standard_datetime = ""
    contributing = False
    contributed_list = []
    for row in list(zipped_data):
        if(row[1]!=standard_adid):
            standard_adid=row[1]
            contributing = False
        if(row[3]=='conv'):
            standard_datetime=row[2]
            contributing = True
        if(contributing):
            if((standard_datetime-row[2]).total_seconds()>=86400*lookback_window):
                contributing = False
            else:
                contributed_list.append(row[0])
    
    data.loc[contributed_list,'contributed']=1

    #df를 다시 뒤집는 코드
    data = data.reindex(index=data.index[::-1])
    data = data.reset_index(drop=True)
    
    #이제 contributed 열에는 conversion과 그 lookback_window에 포함된 경우 1, 아니면 0.
    
    #delete_abusing이 true일 경우, abusing_threshold를 기준으로 걸러냄.
    if(delete_abusing):
        #일단 adid_same을 만들어보자
        df['adid2']=df['adid'].shift(1)
        df['adid_same']=df['adid']==df['adid2']
        #zip()으로 range(len(df)), adid_same, datetime, partner_index들만 존재하는 튜플을 만들어보자!
        #zip은 튜플로 구성된 iterator를 반환해준다.
        zipped_one = zip(range(len(df)),df['adid_same'],df['datetime'])
        start = True
        standard_datetime = ""
        delete_index = []
        for row in (list(zipped_one)):
            if(start):
                standard_datetime = row[2]
                start = False
            else:
                if(row[1]):
                    if((row[2]-standard_datetime).total_seconds()<abusing_threshold):
                        delete_index.append(row[0])
                    else:
                        standard_datetime = row[2]
                else:
                    standard_datetime = row[2]
        df.drop(delete_index, inplace=True)

    #이제 data를 null_path를 구할 용도와 conversion_path를 구할 용도로 나누어보자.
    
    

    ####여기서부터 null로 끝나는 path들을 만드는 과정####
    data_zero = data.copy()

    #1인 행들을 제거하는 코드
    indexNames = data_zero[data_zero['contributed'] == 1 ].index
    data_zero.drop(indexNames , inplace=True)

    #인덱스를 다시 매겨보자
    data_zero = data_zero.reset_index(drop=True)

    #adid_same을 만들어보자. 이전 행과 adid가 일치하는 경우 True 값을 갖게 됨.
    data_zero['adid2']=data_zero['adid'].shift(1)
    data_zero['adid_same']=data_zero['adid']==data_zero['adid2']

    #zip()으로 range(len(df)), adid_same, datetime, partner_index들만 존재하는 튜플을 만들어보자!
    #zip은 튜플로 구성된 iterator를 반환해준다.
    zipped_zero = zip(range(len(data_zero)),data_zero['adid_same'],data_zero['datetime'], data_zero['partner_index'])

    #새로운 dataframe을 만들자. 얘는 나중에 null로 끝날 패스들의 행을 모두 갖게 됨
    columns = ['path', 'conversion','null', 'amount']
    result_zero = pd.DataFrame(columns=columns)

    #session이 언제 나뉘어야 할까?
    # 1. adid가 전행과 다를 때
    # 2. 기준점으로부터 시간이 날짜수*86400초를 넘었을 때

    #이 null_paths에는 생성된 모든 패스들이 생성된 순서대로 담기게 된다. 이후 result_zero의 'path' 열에 바로 붙일 것임!
    null_paths = []

    #start, standard_datetime, path_list는 아래 코드를 돌리기 위해 외부에 설정해놓은 변수
    start = True
    standard_datetime = ""
    path_list = []
    for row in (list(zipped_zero)):
        if(start):#아예 첫 행
            standard_datetime = row[2]
            path_list.append(row[3])
            start = False
        else:#그 외의 행
            #adid가 같은 경우
            if(row[1]):
                #룩백 윈도우를 넘은 경우
                if((row[2]-standard_datetime).total_seconds()>=86400*lookback_window):
                    path = ''.join(path_list)
                    path_list = []
                    path_list.append(row[3])
                    null_paths.append(path)
                    standard_datetime = row[2]
                else:
                    path_list.append(' > ')
                    path_list.append(row[3])
            #adid가 바뀐 경우 : 당연히 세션이 새로 시작되어야 함
            else:
                path = ''.join(path_list)
                path_list = []
                path_list.append(row[3])
                null_paths.append(path)
                standard_datetime = row[2]

    result_zero['path']=null_paths
    result_zero['conversion']=0
    result_zero['null']=1
    result_zero['amount']=0


    ####여기서부터 3초 이하인 것들을 abusing으로 판단해서 제거함####


    ####여기서부터 conversion으로 끝나는 path들을 만드는 과정임####

    data_one = data.copy()

    #0인 행들을 제거하는 코드
    indexNames = data_one[data_one['contributed'] == 0 ].index
    data_one.drop(indexNames , inplace=True)

    #df를 뒤집어보자
    data_one = data_one.iloc[::-1]

    #인덱스를 다시 매겨보자
    data_one = data_one.reset_index(drop=True)

    #adid_same을 만들어보자
    data_one['adid2']=data_one['adid'].shift(1)
    data_one['adid_same']=data_one['adid']==data_one['adid2']

    #zip()으로 range(len(df)), adid_same, datetime, partner_index들만 존재하는 튜플을 만들어보자!
    #zip은 튜플로 구성된 iterator를 반환해준다.
    zipped_one = zip(range(len(data_one)),data_one['adid_same'],data_one['datetime'], data_one['partner_index'])

    #새로운 dataframe을 만들자. columns는 위와 동일하니 그대로.
    result_one = pd.DataFrame(columns=columns)

    conversion_paths = []
    amounts = []

    #session이 언제 나뉘어야 할까?
    # 1. adid가 전행과 다를 때
    # 2. 기준점으로부터 시간이 날짜수*86400초를 넘었을 때

    start = True
    standard_datetime = ""
    path_list = []
    limit = len(data_one)
    for row in (list(zipped_one)):
        if(row[3]=='conv'):
            standard_datetime = row[2]
            last = True
            tmp_index = row[0]+1
            if(tmp_index==limit):
                break
            while(True):
                if(data_one['adid_same'][tmp_index]):
                    if((standard_datetime-data_one['datetime'][tmp_index]).total_seconds()<86400*lookback_window):
                        if(data_one['partner_index'][tmp_index]!='conv'):
                            if(last):
                                path_list.append(data_one['partner_index'][tmp_index])
                                last = False
                            else:
                                path_list.insert(0, ' > ')
                                path_list.insert(0, data_one['partner_index'][tmp_index])
                        tmp_index = tmp_index + 1
                        if(tmp_index == limit):
                            break
                        continue
                    else:
                        break
                else:
                    break
            if(len(path_list)>0):
                conversion_paths.append(''.join(path_list))
                amounts.append(data_one['amount'][row[0]])
                path_list = []

    result_one['path']=conversion_paths
    result_one['amount']=amounts
    result_one['conversion']=1
    result_one['null']=0

    #이제 result_zero와 result_one을 합치자.
    result_complete = pd.concat([result_zero, result_one], ignore_index=True)
    result_complete = result_complete.rename(columns = {'path': 'path', 'conversion': 'total_conversions', 'null': 'total_null', 'amount': 'total_conversion_value'})

    result_complete.to_csv(result_data_name, encoding = 'utf-8', index = False)
