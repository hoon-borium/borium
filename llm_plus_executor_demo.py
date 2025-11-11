#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LLM + 실행기(파이썬) 데모
# - LLM은 '질문 의도'를 JSON으로만 출력하고,
#   실제 데이터 집계/비교는 파이썬(Pandas)이 수행합니다.
# - 정확도/재현성/감사성(근거) 확보에 유리합니다.
#
# 사용법:
#   python llm_plus_executor_demo.py --file /path/to/data.xlsx --model llama3.2 --question "지난주 일평균 대비 어제 매출은?"
import argparse
import json
import sys
import subprocess
import pandas as pd
from zoneinfo import ZoneInfo

def pick_column(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return cols[0]

def load_table(path):
    if path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    date_col = pick_column(df.columns, ['date','날짜','일자','order_date','created_at'])
    amount_col = pick_column(df.columns, ['amount','매출','매출액','sales','sales_amount','revenue'])
    df = df[[date_col, amount_col]].copy()
    df.columns = ['date','amount']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df

JSON_SCHEMA_DESC = '''{
  "intent": "aggregate | compare | topN | trend | breakdown",
  "metric": "sales_amount",
  "time": {
    "a": "yesterday | last_week | last_week_avg | last_7d | <YYYY-MM-DD>",
    "b": "same options, optional"
  }
}'''

PROMPT_TMPL = '''SYSTEM:
너는 판매 데이터 질문을 "파라미터 JSON"으로만 변환하는 추출기다.
반드시 한 줄 JSON만 출력하고, 다른 텍스트는 출력하지 마라.

스키마:
{schema}

예시:
Q: "지난주 일평균 대비 어제 매출은?"
A: {{"intent":"compare","metric":"sales_amount","time":{{"a":"yesterday","b":"last_week_avg"}}}}

USER 질문:
{question}
'''

def call_ollama_json(model, prompt):
    proc = subprocess.run(['ollama','run',model], input=prompt.encode('utf-8'),
                          capture_output=True, check=True)
    text = proc.stdout.decode('utf-8', errors='ignore').strip()
    # 코드블록 제거
    if text.startswith('```'):
        text = text.strip('` \n')
        if '\n' in text:
            text = text.split('\n',1)[1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            s = text[text.index('{'): text.rindex('}')+1]
            return json.loads(s)
        except Exception:
            raise ValueError(f'모델 JSON 파싱 실패: {text}')

def compute_compare_y_vs_lastweekavg(df):
    tz = ZoneInfo('Europe/London')
    today = pd.Timestamp.now(tz).normalize()
    yday = (today - pd.Timedelta(days=1)).date()

    a_df = df[df['date'].dt.date == yday]
    a = a_df['amount'].sum() if not a_df.empty else None

    y_ts = pd.Timestamp(yday, tz=tz)
    last_week_end = (y_ts - pd.Timedelta(days=(y_ts.weekday()+1)%7)).normalize()
    last_week_start = last_week_end - pd.Timedelta(days=6)
    lw = df[(df['date']>=last_week_start) & (df['date']<=last_week_end)]
    b = lw.groupby(lw['date'].dt.date)['amount'].sum().mean() if not lw.empty else None

    return a, b, (last_week_start.date(), last_week_end.date())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--model', default='llama3.2')
    ap.add_argument('--question', required=True)
    args = ap.parse_args()

    df = load_table(args.file)
    if df.empty:
        print('데이터가 비어있거나 날짜 파싱 실패')
        sys.exit(1)

    prompt = PROMPT_TMPL.format(schema=JSON_SCHEMA_DESC, question=args.question)
    try:
        parsed = call_ollama_json(args.model, prompt)
    except subprocess.CalledProcessError as e:
        print('Ollama 호출 오류:', e.stderr.decode('utf-8', errors='ignore'))
        sys.exit(2)
    except Exception as e:
        print(str(e)); sys.exit(3)

    print('=== LLM JSON ===')
    print(json.dumps(parsed, ensure_ascii=False))

    # 데모: compare(yesterday, last_week_avg)만 지원
    if parsed.get('intent') == 'compare' and parsed.get('metric') in ['sales','sales_amount']:
        t = parsed.get('time', {})
        if t.get('a') == 'yesterday' and t.get('b') in ['last_week_avg','lastweek_avg','last_week_average']:
            a, b, (s,e) = compute_compare_y_vs_lastweekavg(df)
            if a is None or b is None or pd.isna(b):
                print('데이터 부족으로 계산 불가')
                sys.exit(0)
            diff = a - b
            pct = (diff / b)*100 if b != 0 else None
            sign = '증가' if diff>0 else '감소' if diff<0 else '변동 없음'
            print('=== 결과 ===')
            print(f'어제 매출: {a:,.0f}원')
            print(f'지난주({s}~{e}) 일평균: {b:,.0f}원')
            print(f'차이: {abs(diff):,.0f}원 {sign}' + (f' ({pct:.1f}%)' if pct is not None else ''))
            print('\n[근거] 정의: 어제=현재(Europe/London) 기준 D-1, 지난주=직전 월~일, 일별 합계의 평균')
            sys.exit(0)

    print('아직 이 JSON 조합은 데모 실행기에 없어요. (intent/시간 프리셋 추가 필요)')

if __name__ == '__main__':
    main()
