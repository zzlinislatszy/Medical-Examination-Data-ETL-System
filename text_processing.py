import os
from datetime import datetime
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from utils import log_execution_time
from data_preprocessing_251029 import postprocess_multilang
from db_to_dataframe_251029 import db_to_dataframe
from llm_processing_251029 import process_suggestion
from fastapi import APIRouter, HTTPException

router = APIRouter()

# 各語系對應的欄位子集合
SUBSET = {
    '1':['RECORD_ID', 'LANG_NO', 'GROUPNO', 'TCNAME_GROUP', 'ITEM_CODE', 'TCNAME_ITEM', 'COMMENT', 'TCNAME_SUMMARY'],
    '2':['RECORD_ID', 'LANG_NO', 'GROUPNO', 'ENNAME_GROUP', 'ITEM_CODE', 'ENNAME_ITEM', 'ENNAME_COMMENT', 'ENNAME_SUMMARY'],
    '3':['RECORD_ID', 'LANG_NO', 'GROUPNO', 'JPNAME_GROUP', 'ITEM_CODE', 'JPNAME_ITEM', 'JPNAME_COMMENT', 'JPNAME_SUMMARY'],
    '4':['RECORD_ID', 'LANG_NO', 'GROUPNO', 'SCNAME_GROUP', 'ITEM_CODE', 'SCNAME_ITEM', 'SCNAME_COMMENT', 'SCNAME_SUMMARY']
}

# 各語系預設文字對照表
LANGU_MAP = {
    '1': '本項無補充說明',
    '2': 'No additional information for this item.',
    '3': 'この項目に関する追加情報はありません。',
    '4': '本项无补充说明。'
}


# 依序將 RECORD_ID_LST 中的 record_id，從 preprocessed_df 擷取，整併為可讀文本
@ log_execution_time
def text_processing(preprocessed_df: pd.DataFrame, processed_report_csv_path: Optional[str], api_requests: List[Dict[str, Any]]) -> pd.DataFrame:
    text_processed_rows = []

    record_groups = preprocessed_df.groupby('RECORD_ID', sort=False)
    for api_request in api_requests:
        record_id = api_request['RECORD_ID']
        df = record_groups.get_group(record_id)
        langu_no = str(df['LANG_NO'].iloc[0]).strip()         # 取得該 record 的 LANG_NO -> 1 / 2 / 3 / 4
        df = df[SUBSET[langu_no]]                             # 依 LANG_NO 選取對應欄位

        # rename col 為通用名稱
        report_df =  df.rename(columns={
            f'{SUBSET[langu_no][3]}': 'GROUP',
            f'{SUBSET[langu_no][5]}': 'ITEM_NAME',
            f'{SUBSET[langu_no][6]}': 'COMMENT',
            f'{SUBSET[langu_no][7]}': 'SUMMARY'
        })

        target_json = next((item for item in api_requests if item["RECORD_ID"] == str(record_id)), None)
        target_json = json.dumps(target_json, ensure_ascii=False) if target_json else ''

        output = process_1_record(langu_no, report_df)
        text_processed_rows.append([str(record_id), output, target_json])

    df_out = pd.DataFrame(text_processed_rows, columns=['record_id', 'report', 'request'])

    if processed_report_csv_path:
        df_out.to_csv(processed_report_csv_path, index=False)

    return df_out


# 同一 record 的記錄整併為層次化文字輸出
def process_1_record(langu_no: str, report_df: pd.DataFrame) -> str:
    """
    層級關係：
    GROUP                      1st
        ITEM                   4th
            COMMENT            3rd
                SUMMARY        2nd 原文
                SUMMARY        2nd 改寫後
    """
    summary_2_llm = [s.strip() for s in report_df['SUMMARY'].drop_duplicates().to_list() if s]
    summary_translated = process_suggestion(langu_no, summary_2_llm, mode='azure', model=os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o'))

    lines = []

    for group, df_group in report_df.groupby('GROUP', sort=False):
        lines.append(group.strip())

        summary_blocks = []
        for summary, df_summary in df_group.groupby('SUMMARY', sort=False):

            if summary == LANGU_MAP[langu_no]:
                for comment, df_comment in df_summary.groupby('COMMENT', sort=False):
                    item = get_unique_item_names(df_comment)
                    summary_blocks.append({
                        'items': item,
                        'comments': [comment.strip()],
                        'summary': summary.strip()
                    })
            else:
                item = get_unique_item_names(df_summary)
                comments = df_summary['COMMENT'].str.strip().drop_duplicates().to_list()
                summary_blocks.append({
                    'items': item,
                    'comments': comments,
                    'summary': summary.strip()
                })

        first_seen = {}
        for i, b in enumerate(summary_blocks):
            it_key = tuple(b['items'])
            if it_key not in first_seen:
                first_seen[it_key] = i

        indexed_blocks = list(enumerate(summary_blocks))
        indexed_blocks.sort(key=lambda t: (first_seen[tuple(t[1]['items'])], t[0]))

        last_it_key = None
        for _, block in indexed_blocks:
            it_key = tuple(block['items'])
            if it_key != last_it_key:
                lines.append(f"    {'、'.join(block['items'])}")
                last_it_key = it_key

            if block['comments']:
                lines.append(f"        {'、'.join(block['comments'])}")

            summary = block['summary']
            if summary:
                lines.append(f"            {summary_translated.get(summary, summary)}\n")

    return '\n'.join(lines)


def get_unique_item_names(df_summary: pd.DataFrame) -> list:
    item_dict = {}
    result = []

    for _, row in df_summary.iterrows():
        item_code = str(row['ITEM_CODE']).strip()
        item_name = str(row['ITEM_NAME']).strip()

        if item_code not in item_dict:
            item_dict[item_code] = item_name
            result.append(item_name)

    return result


@router.post("/process")
def process_api(api_requests: Any):
    """
    接收 api_request，處理流程如下：
    db_to_dataframe -> postprocess_multilang -> 由 df_unique 取得 record_id -> text_processing
    並回傳 text_processing 之 df_out(JSON)。
    """
    try:
        api_requests = [api_requests] if isinstance(api_requests, dict) else api_requests

        final_df = db_to_dataframe(api_requests)
        preprocessed_df = postprocess_multilang(final_df)

        output_dir = './output_02_text_processed'
        data_dir = './output_01_preprocessed'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%y%m%d_%H%M')
        data_path = os.path.join(data_dir, f'data_processed_{timestamp}.csv')
        csv_path = os.path.join(output_dir, f'text_processed_{timestamp}.csv')

        preprocessed_df.to_csv(data_path, index=False)

        df_out = text_processing(
            preprocessed_df=preprocessed_df,
            processed_report_csv_path=csv_path,
            api_requests=api_requests,
        )

        return {"rows": df_out[['report']].to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
