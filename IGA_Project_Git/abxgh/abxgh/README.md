> 사실 리드미는 아니구 프로젝트 참여자를 위해 전체 흐름을 정리하는 문서입니다.
> last edit: 2019-11-22

## 프로젝트의 목적
본 프로젝트의 목표는 iga 데이터에 data-driven attribution model을 적용하는 솔루션을 제작하는 것이다.
- 최소한의 목표: 전처리 포함, 이번에 제공된 이커머스 데이터에 데이터 드리븐 모델을 적용하여 기여도 산출하기
- 회사 입장에서 바라는 점: 코드의 패키지화(=파라미터 다양하게 적용 가능, 새로운 데이터가 들어왔을 때 바로 적용 가능)
11월 3주까지 최소한의 목표 달성. 11월 4주 패키지화 달성. 11월 4주 이후 새로운 목표 돌입

## Attribution 이론

> 들어가기 전에! 용어 정리  
> 고객이 광고를 본 경험에 대해서 터치포인트, 채널, 매체 등 다양한 용어가 혼재해서 사용되는데, iga 데이터에는 `partner_key`에 해당하는 value들이 최종적으로 기여도를 할당받는 매체를 가리킨다. iga 데이터는 현재 고객이 클릭한 데이터만 집계 가능하며, `partner_key`는 페이스북, 구글과 같은 매체 종류에 해당한다(광고의 종류, 캠페인의 종류가 아님)

<details close>
    <summary>TMI: 왜 이런게 있나</summary>
    디지털 미디어의 확산으로 고객이 구매에 도달하는 과정이 다변화되었다. 이전과 달리 다양한 채널과 디바이스로 고객이 브랜드에 노출되다 보니 고객 여정이 파악하기 어려워졌다. 이는 전과 달리 각 매체의 광고 효과를 제대로 측정하기 어렵다는 뜻이기도 하다. 전환 경로에 따라 각 접접(터치포인트)이 전환을 달성하는데 얼마나 기여했는지 알 수 있는 공식적인 프로세스가 필요해졌다. 이 프로세스를 attribution이라 한다.
    <br>
    어트리뷰션 모델은 기여도를 분배하는 매체의 수에 따라 single touch/multi touch로 나뉘며, 방법론 상에 따라 heuristic과 data-driven으로 나뉜다. 본 프로젝트에서는 방법론에 따른 구분을 택한다.
</details>
<br>
<details close>
    <summary>어트리뷰션 모델 전반 관련 링크</summary>
    <ul>
        <li> 찬영 제작 intro 슬라이드 https://drive.google.com/drive/folders/1OxFElHVgS-1ynJ2WQbP-jhzhri3_4S8O </li>
        <li> 전체 첫 회의 때 공유했던 introduction 문서(검토했던 모든 데이터드리븐 모델에 대한 설명 포함) https://docs.google.com/document/d/1SzT0l97ICk8fGAJn1xMvFZLIkeKr4a0uJObomtRC514/edit </li>
        <li>
        AB180 왜 라스트터치가 아니라 샤플리인가 http://blog.ab180.co/data-science-with-r-4-multi-touch-attribution/ </li>  
    </ul>
</details>



### 휴리스틱 모델
다양한 휴리스틱 모델이 존재하지만 여기서는 실제로 코드로 구현할 예정인 모델만 요약한다
- last touch: 마지막 매체에만 모든 크레딧 부여. 고객과 브랜드가 가장 최근에 접한 채널을 중요시. 어떤 매체가 즉각적인 전환을 끌고오는가를 파악하는 데 효과적
- first touch: 고객과 브랜드가 처음 만난 매체에 모든 크레딧 부여
- time decay: 전환에 가까이 이르렀을 때 비례하여 크레딧이 증가됨
- linear: 모든 접점에 동일하게 크레딧 부여
- position based: U shape, W shape 등 사용자 설정 가중치에 따라 크레딧 부여

### 데이터 드리븐 모델
bagged logistic model이나 simple probabilistic 모델 등 다양한 모델을 고려했지만, 현업에서 가장 널리 쓰이는 두 모델을 구현하기로.

#### 사례로 설명하는 두 모델의 차이점  
아래 표에서 b와 c에 주목했을 때, 샤플리 모델이라면 c가 기여도가 높으며, 마르코프 모델이라면 b가 기여도가 높다

|  path | total_conversions | total_conversion_value |
|:-----:|-------------------|------------------------|
| a > b | 1                 | 5,000                  |
| a > c | 1                 | 13,000                 |
| a > b | 1                 | 5,000                  |

#### shapely
- 장점: 게임이론에 기반하여 하나의 사건에 대해 단일 성과의 크기를 플레이어들이 분배하는 모델이기 때문에 conversion이 초래한 수익을 성과로서 고려할 수 있다. 관심 있는 전환이 판매이고, 단순히 '구매했다/구매안했다'가 아니라 '얼마나 구매했는가'를 보고자한다면 적절한 모델
- 단점:
1. 플레이어들의 순서가 고려되지 않고 순서 없는 조합을 따진다. 광고의 사례에서 A > B 와 B > A는 샤플리모델에서 같은 사건이다. 이는 기존 휴리스틱 모델들이 대부분 순서를 중점적으로 봤던 관례와 대치되는 사고방식.
2. 매체를 클릭한 빈도수를 고려하지 않는다. 예를 들면 샤플리 모델에서 A > B > A 와 A > B 는 같은 사건이다.

#### markov
<details close>
    <summary>마르코프 모델 관련 링크</summary>
    - R로 마르코프 모델 구축하는 블로그글(리무벌 이펙트 계산 과정에 대한 자세한 설명) https://analyzecore.com/2016/08/03/attribution-model-r-part-1/
</details>
  
- 장점: 고객이 거쳐 간 매체를 순서대로 고려하며 이행 확률을 도출한다는 점에서 직관적이다.
- 유의: 마르코프 모델에서 이행은 인과와는 관계가 없다.
- 단점: 마르코프 모델은 상태와 상태 간 이행을 설명하는 모델이기 때문에 구매액에 따른 가중치를 줄 수 없다. 마지막 상태는 전환이 된 상태이거나 전환이 되지 않은 상태 둘 중에 하나다.


## 데이터 이해하기
### 데이터의 구조
1. adtouch 데이터: 광고 클릭 데이터. 스크롤해서 넘어간 경우(impression)는 기록되지 않으며, 클릭한 경우만 집계되고 있다.
2. event 데이터: 앱 내에서 사용자의 활동 데이터
3. attribution 데이터: (1차 모델에는 활용되지 않음) 애플, 구글, 페이스북 등 대부분의 큰 기업들은 url을 통해 iga에서 광고 클릭을 트래킹하는 것을 막아놨다. 그래서 애드터치 데이터에는 해당 기업에 대한 광고 클릭은 부분적으로 집계? 집계되지 않고? 있으며, 매체 측에서 가공한 데이터를 요청해서 제한적인 형태로 가져온 것이 attribution data. 여기에는 클릭 시점을 정확히 알 수 없고 매체가 결정한 룩백 윈도우에 따라 conversion에 기여했는지/안했는지만 알 수 있다.

### 1차 모델 주요 변수
#### adtouch
- adid: 어뷰저처리를 건너뛰고 하려 했으나, 한 달에 100회, 200회 등 비정상적으로 클릭 수가 많은 adid가 존재하여 __빈도수 하위 90%__ 만 분석에 포함시키기로함. 1차적으로 핑거프린트나 ip는 고려하지 않으며 adid를 유일한 식별자로 이용한다.
- partner_key: 광고 매체. 앞선 방식에 따라 adtouch 데이터를 한 번 거르면 한 달에 10개 이내의 고유한 파트너 키가 존재한다.
- server_datetime: adtouch 데이터의 기준 시각
   <details close>
    <summary>쓸 수 없는 변수들</summary>
    <ul>
        <li> `utm_source`, `utm_medium`: 파트너 키와 관련있지 않을까 생각했으나 대부분 비어있음 </li>
        <li> `campaign_id`: 파트너 키보다 상위 수준의 캠페인 단위 id. 최근부터 적용하기 시작하여 과거 데이터에는 쓸 수 없다 </li>
    </ul>
</details>

#### event
- adid: 애드터치 데이터와 매칭할 adid
- event_datetime: event 데이터의 기준 시각. 데이터 내에 여러 datetime이 존재하는데 event_datetime → request_datetime → server_datetime 순서로 찍히며, event_datetime은 correction_datetime와 같다. 이 중 애드터치 데이터와 가장 가까운 것(=가장 먼저 찍히는 시각)이므로 event_datetime으로 정했다.
- event_name: 우리가 보고자 하는 것은 purchase. cf)  start_session은 무조건 처음에 찍히고, 그 다음에 특별한 session에 대해서만 firstopen, daily_first_open, deeplinkopen 등이 찍히는 것


## 데이터 전처리
[참고]
- (수연이 전처리 리서치 결과 구글 독스)[https://docs.google.com/document/d/12cIO1T7-3Yv-Hd26eJL8YTB6HoUlXaKfyx9X7zCbg0s/edit]
- (수연이 전처리 원문 medium 링크)[https://medium.com/data-from-the-trenches/marketing-attribution-tutorial-part-2-7bb78dec502]
   <details close>
    <summary>에반의 path 구축 접근법</summary>
    우선 터치를 찾을 이벤트는 정해져 있습니다. 이벤트 데이터에 event_name 이 abx:first_open과 abx:deeplink_open 입니다.
    <br>
    첫번째 abx:first_open일 때
    1. adid
    2. finger_print
    3. ip
    4. referrer 정보(이건 추후 데이터 전달할 때 추가 해서 전달 드리겠습니다.) 매칭은 터치데이터(key) 와 adkey
    <br>
    두번째 abx:deeplink_open는 터치데이터(key)와 deeplink_payload안에 adkey 정보
    궁금한 점 있으면 말씀주세요.
</details>

 원래는 에반의 답변에 따라 event 데이터 내의 first_open 혹은 deeplink_open 과 이 open을 촉발시킨 adtouch 데이터의 광고 클리을 매칭하는 식으로 접근하려 했다. 이 때 사용되는 것이 event 데이터의 `referrer`(first_open의 경우)와 `deeplink_payload`(deeplink_open의 경우)라는 변수. 각 변수에 들어있는 정보를 활용해 데이터의 매체 관련 정보까지 일치해야 해당 클릭이 해당 오픈을 발생시켰다고 볼 수 있다는 것이다.  
 그러나 우리의 목표는 이렇게 직접적으로 촉발한 광고를 포함하여 그 이전에 영향을 미치는 광고도 포함하여 path를 구축하는 것이므로(즉, 수연이의 링크에 따르면 micro session이 아니라 macro 세션을 구축해야하므로) conversion 이벤트에 대하여 adid가 일치하고 timestamp가 이벤트보다 이전인 광고들을 어차피 모두 봐야한다. 즉, **매체 매칭은 불필요하며 구매를 완료한 사용자(adid)가 그 이전에(timestamp) 어떤 매체에서 광고를 클릭했는지**가 우리의 관심사다. 그 중 어떤 클릭을 path로 할 것인가는 다음 두 가지 방법을 시도해보았다.

### lookback_window 기준으로 path 나누기
메인 함수에 들어가는 parameter 중 raw_data는 전처리 과정을 거쳐 adid, datetime, partner_key, event_name, amount, partner_index의 column을 가지는 table로써 만들어집니다. 먼저, 'lookback_window'라는 날짜 이내에 발생한 touch는 conversion에 기여한 것으로 계산해서 contributed을 1과 0으로 표시해줍니다. 그 후, path는 partner_key를 인덱스화해서 연결해줍니다. 모든 path는 conversion 혹은 conversion 없이(null) 마무리되며, 양쪽 path 모두 lookback_window를 기준으로 해서 일정 단위로 구분해줍니다. 이후 conversion과 null로 마무리된 세션들을 union해서 path, total_conversion, total_null, total_conversion_value(구매액)을 가지는 table로 저장해서, 각 path별로 conversion/null 여부와 구매액을 포함하고 있습니다.

* 변경할 수 있는 parameter
lookback_window: 클수록 path가 길어지며, 구매가 발생했을 때 이전의 광고가 그 구매에 영향을 주었다고 간주할 기간을 의미합니다.  일(day) 단위로 입력되며, 코드 내에서 초 단위로 환산되어 활용됩니다.
raw_data_name: 상기에서 언급한 coloumn들을 포함하는 csv파일 이름으로, input data의 이름입니다.
result_data_name: 결과로써 저장되는 파일로 상기에서 언급한 column들을 포함합니다.

### inactivity 기준으로 path 나누기
'inactivity'라는 시간 단위를 기준으로 path를 생성하였고, path는 partner_key를 인덱스화하여 연결한 것입니다.
3번의 전처리 과정을 끝마치면 최종 전처리 완료된 컬럼은 path, total_conversions, total_conversion_value, total_null 입니다.
첫번째 전처리로는 path를 만들기 위해 partner_key를 인덱스화하여 만들었으며, lookback_window 팀과 전처리 작업을 일치시키기 위해 공통된 partner_key만 사용하였습니다.
두번째 전처리로는 'micro_session'을 만들었으며, 이는 기준으로 정한 'inactivity'보다 'timediff'가 작다면 0으로 반환하여 한 path로 묶어 주는 칼럼입니다.
마지막으로는 위의 두 과정을 통해 반환된 데이터프레임을 path별로 total_conversions, total_conversion_value, total_null 값을 계산하여 반환하는 데이터프레임으로 변환합니다.


## 모델 적용
### 레퍼런스 선정
우리를 스쳐지나간 코드들
### 구조
나중에 추가
### markov logic

### shapely logic
