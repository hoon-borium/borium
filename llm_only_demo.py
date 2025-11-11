#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LLM 단독 데모 (Ollama + 텍스트 데이터로만 추론)
# - 아이디어: 엑셀/CSV에서 최근 N일 데이터를 텍스트로 추려 모델에 그대로 넣고,
#   '지난주 일평균 대비 어제 매출' 같은 질문을 모델에게 직접 계산시킵니다.
# - 정확도/재현성은 보장하지 않습니다. '끝까지 돌아가는 루프' 데모용입니다.
#
# 사용법:
#   python llm_only_demo.py --file /path/to/data.xlsx --model llama3.2 --days 14 --question "지난주 일평균 대비 어제 매출은?"
import argparse
import sys
import subprocess
import pandas as pd
from io import StringIO

def pick_column(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # fallback
    return cols[0]

def load_table(path):
    if path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # heuristic: find date and amount
    date_col = pick_column(df.columns, ['date','날짜','일자','order_date','created_at'])
    amount_col = pick_column(df.columns, ['amount','매출','매출액','sales','sales_amount','revenue'])
    df = df[[date_col, amount_col]].copy()
    df.columns = ['date','amount']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df

def to_csv_text(df, days):
    df = df.sort_values('date')
    if not df.empty:
        end = df['date'].max().normalize()
        start = end - pd.Timedelta(days=days-1)
        df = df[(df['date'] >= start) & (df['date'] <= end)]
    tmp = df.copy()
    tmp['date'] = tmp['date'].dt.strftime('%Y-%m-%d')
    out = StringIO()
    tmp.to_csv(out, index=False)
    return out.getvalue()

PROMPT_TMPL = '''SYSTEM:
너는 데이터 분석 LLM이야. 아래의 CSV만 보고 사용자의 질문에 답해.
반드시 수치를 추론해서 한국어로 간단히 설명해. 불확실해도 추정해서 말해도 된다.

CSV(최근 데이터 일부):
{csv_text}

USER 질문:
{question}
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Excel or CSV file path')
    ap.add_argument('--model', default='llama3.2')
    ap.add_argument('--days', type=int, default=14, help='recent N days to include')
    ap.add_argument('--question', required=True)
    args = ap.parse_args()

    df = load_table(args.file)
    if df.empty:
        print('데이터가 비어있거나 날짜 파싱 실패')
        sys.exit(1)
    csv_text = to_csv_text(df, args.days)

    prompt = PROMPT_TMPL.format(csv_text=csv_text, question=args.question)
    print('=== LLM 요청 프롬프트(요약) ===')
    print(prompt[:1000] + ('...\n' if len(prompt) > 1000 else '\n'))
    print('=== 모델 응답 ===')
    try:
        proc = subprocess.run(['ollama', 'run', args.model], input=prompt.encode('utf-8'),
                              capture_output=True, check=True)
        print(proc.stdout.decode('utf-8', errors='ignore'))
    except subprocess.CalledProcessError as e:
        print('Ollama 호출 오류:', e.stderr.decode('utf-8', errors='ignore'))
        sys.exit(2)

if __name__ == '__main__':
    main()
