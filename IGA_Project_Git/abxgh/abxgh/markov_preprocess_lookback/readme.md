# 이 문서는 다음 두 파일에 대한 설명을 다루고 있습니다.
## 'IGA_lookback_complete.ipynb', 'lookback_data.csv'

# 'IGA_lookback_complete.ipynb'에 대한 설명
## 'lookback_data.csv'를 읽어들여, 'lookback_path_complete.csv'를 생성합니다.
- 'lookback_data.csv'는 'adid, datetime, partner_index, amount' 의 네 column을 가진 데이터테이블입니다.
- 이용자들을 구분하는 단위인 adid, 이용자의 활동이 벌어진 시간인 datetime, 이용자의 활동의 종류를 구분하는 partner_index, 이용자의 활동 중 '구매'의 경우에 그 액수를 의미하는 amount의 정보를 담고 있습니다.
- partner_index에 해당하는 활동은 고객이 구매에 도달했을 경우인 conv와, 그 외 노출된 광고의 종류로 구분한 광고노출의 경우(a, b, c, ...)입니다.
- 중간 단계에서 'lookback_data_marked_one.csv'를 생성, 저장합니다. 이는 후의 코드에서 다시 사용하게 됩니다.
- 'lookback_path_complete.csv'는 'path, conversion, null, amount' 의 네 column을 가진 데이터테이블입니다.
- path는 고객이 구매(conversion)에 도달했는지의 여부에 따라 두 종류로 나뉠 수 있습니다.
    1. 이용자가 conversion에 도달한 경우
       그로부터 과거 방향으로 lookback window를 적용하였을 때 포함되는 광고노출을 노출 순서에 따라 이어붙인 문자열이 path가 됩니다.
       고객이 a, b, c 순서로 광고에 노출 된 후 conversion에 도달해 18000의 amount의 구매를 행했을 경우
       lookback_path_complete.csv에는 {'path': 'a > b > c', 'conversion': 1, 'null': 0, 'amount': 18000}의 행이 추가됩니다.
    2. 이용자가 conversion에 도달하지 않은 경우
       한 이용자가 a, b, c, d 순서로 광고에 노출되었고 a, b, c, d의 노출 모두 lookback window에 포함되지 않은 경우
       a를 기준으로 미래 방향으로 lookback window를 적용하였을 때 포함되는 광고노출을 노출 순서에 따라 이어붙인 문자열이 path가 됩니다.
       해당 path에 포함되지 못한 노출들 중 첫 노출이 다음 path의 시작이 됩니다.
       만약 위의 예시에서 a, b, c가 conversion에 도달하지 못한 한 path를 이루게 된다면,
       lookback_path_complete.csv에는 {'path': 'a > b > c', 'conversion': 0, 'null': 1, 'amount': 0}의 행이 추가됩니다.

## 코드 내용
### 코드의 작동 방식에 대한 전반적인 설명은 주석으로 다 해 두었습니다.
### 추후 이곳에 기술해야 한다고 판단되면 추가하도록 하겠습니다.

## 수정이 필요한 사항
- 우선 코드가 상당히 느립니다. 한 달치 데이터를 분석하는 데에 4분 정도가 걸립니다.
  이는 contributed를 표시하는 과정에서 3분이 소요되는 것으로 미루어 보아, 이 부분을 수정하면 해결될 것 같습니다.