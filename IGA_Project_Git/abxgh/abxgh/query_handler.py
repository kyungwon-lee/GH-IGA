from pyathena import connect
from pyathena.pandas_cursor import PandasCursor

class Handler: 
    def __init__(self, aws_access_key, aws_secret_access_key, 
    region = 'ap-northeast-2', default_dir = 's3://adbrix.gh.data/'):
        
        # TODO: 클래스 variable 추가

        self.cursor = connect(aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_access_key,
                    s3_staging_dir=default_dir,
                    region_name=region,
                    cursor_class=PandasCursor).cursor()

    def filter_adid(self, yearmonth='201908', percentile=0.8):
        """
        애드터치데이터에서 빈도 하위 percentile 만큼 가져오기. 쿼리 실행결과는 pandas DataFrame 형태로 반환
        parameter
        -------
        yearmonth: yyyymm 형식의 string
        percentile: 0에서 1사이의 float
        -------
        output
        -------
        df.adid: 애드아이디. row별 중복 있음
        df.event_time: purchase가 발생한 시각
        df.amount: purchase 당시 결제액. price에 quantity를 곱한 값
        -------
        """
        yyyymm = yearmonth
        percentile = str(percentile)

        filtering_query = """
        SELECT adid, 
            counts 
        FROM 
            (SELECT adid, 
                count(*) counts 
            FROM sampledb.adtouch_table_{0} -- ITERATION: 월별 애드터치 데이터 
            
            GROUP BY  adid ) 

        GROUP BY adid, counts 

        HAVING counts <= APPROX_PERCENTILE(counts, {1}) 
        """.format(yyyymm, percentile)

        df = self.cursor.execute(filtering_query).as_pandas()
        return df

    def filter_purchase(self, yearmonth='201908', percentile=0.8):
        """
        애드터치 데이터에서 빈도 하위 percentile에 해당하는 adid를 추린 뒤, 이들의 purchase 기록을 events 데이터에서 가져오기. 쿼리 실행결과는 pandas DataFrame 형태로 반환
        parameter
        -------
        yearmonth: yyyymm 형식의 string
        percentile: 0에서 1사이의 float
        -------
        ourput

        """
        yyyymm = yearmonth
        percentile = str(percentile)

        purchase_query = """
        -- 특정 카운트 이하로만 등장하는 adid 테이블
        WITH filtered AS (
            SELECT adid, 
                counts 
            FROM 
                (SELECT adid, count(*) counts 
                FROM sampledb.adtouch_table_{0} -- ITERATION: 월별 애드터치 데이터 
                GROUP BY  adid ) 

            GROUP BY adid, counts 
            
            HAVING counts <= APPROX_PERCENTILE(counts, {1})
        )


        -- 해당 adid에 포함되는 adid만 purchase 뽑기
        SELECT adid, event_datetime
            , evt_param_item_price * evt_param_item_quantity as amount
            -- , event_name

        FROM sampledb.events_table_{0} t1 -- ITERATION: 월별 이벤트 데이터
        WHERE event_name = 'abx:purchase'
        and EXISTS (
        SELECT adid
        FROM filtered t2
        WHERE t1.adid = t2.adid
        );
        """.format(yyyymm, percentile)

        df = self.cursor.execute(purchase_query).as_pandas()
        return df

if __name__ == '__main__':
    import config
    import pandas as pd

    handler = Handler(config.Access_key_ID, config.Scret_access_key)
    # print(handler.filter_adid().head())
    for i in ['201908', '201909', '201910']:
        df = handler.filter_purchase(yearmonth=i)
        df.to_csv("data_preprocessing/purchase_"+i+".csv")

        print(df.head())
    

    # 경원 요청 - 이게 아닌가ㅂ다
    # for i in ['201908', '201909', '201910']:
    #     df = handler.filter_purchase(i)
    #     df.to_csv('data_preprocessing/purchase_wtevtname_'+i)