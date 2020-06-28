## 'preprocess_inactivity' 폴더 내 들어 있는 파일 목록
- readme.md
- union_201908.csv ;넣어야 할 csv 파일(쿼리작업 후 뽑은 것)
- IA_201908_without_all_direct.csv ;출력파일
- Final_Inactivity_time.ipynb ;상세한 주석 설명
- pathmaker_inactivity.ipynb ;함수화

-----------------
## readme

이 문서는 다음 두 파일에 대한 설명을 다루고 있습니다.
'Final_Inactivity_time.ipynb', 'IA_201908_without_all_direct.csv'

## 'Final_Inactivity_time.ipynb'에 대한 설명
'union_201908.csv'를 읽어들여, 'IA_201908_without_all_direct.csv'를 생성합니다.

'union_201908.csv'는 'adid', 'datetime', 'partner_key',' event_name', 'amount'의 변수들이 포함되어 있으며,
adid는 이용자별로 부여되는 고유한 id이고, datetime은 해당 adid가 접속한 시간입니다. 
partner_key는 이용자의 활동의 종류를 구분하며, 이용자가 '구매'를 한 경우 event_name은 'abx:purchase'로 기록되며
이것이 'Final_Inactivity_time.ipynb'에서 활용할 conversion을 뜻합니다. 또한 '구매'가 일어날 경우 amount에 그 구매 금액의 정보가 담깁니다.

'Final_Inactivity_time.ipynb'파일을 통해 전처리 과정을 모두 마치면 'IA_201908_without_all_direct.csv' 파일이 생성됩니다.

## 코드 내용
'Final_Inactivity_time.ipynb'는 3번의 전처리 과정을 담고 있으며,
'inactivity'라는 시간 단위를 기준으로 path를 생성하였고, path는 partner_key를 인덱스화하여 연결한 것입니다.
3번의 전처리 과정을 끝마치면 최종 전처리 완료된 컬럼은 path, total_conversions, total_conversion_value, total_null 입니다.

첫번째 전처리로는 path를 만들기 위해 partner_key를 인덱스화하여 만들었으며, lookback_window 팀과 전처리 작업을 일치시키기 위해 공통된 partner_key만 사용하였습니다.
두번째 전처리로는 'micro_session'을 만들었으며, 이는 기준으로 정한 'inactivity'보다 'timediff'가 작다면 0으로 반환하여 한 path로 묶어 주는 칼럼입니다.
마지막으로는 위의 두 과정을 통해 반환된 데이터프레임을 path별로 total_conversions, total_conversion_value, total_null 값을 계산하여 반환하는 데이터프레임으로 변환합니다.

코드의 상세한 작동 방식의 경우, 'Final_Inactivity_time.ipynb'에 주석 처리를 해 두었습니다.

## 소요 시간
해당 한 달 치 데이터를 분석하는 데 소요되는 시간은 1분 미만입니다.

