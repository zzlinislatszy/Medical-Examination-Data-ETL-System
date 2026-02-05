"""
Demo 版：
- 保留資料清洗與缺值處理邏輯
- 為符合保密協議，已移除任何可對應特定業務/系統/代碼之規則
"""
import pandas as pd
from typing import Dict

# 各語系預設對應表：SUMMARY 或 GROUP 缺值時，依語言填入預設值
LANGU_DEFAULT_MAP = {
    '1': {'TCNAME_SUMMARY':'本項無補充說明', 'TCNAME_GROUP': '其他'},
    '2': {'ENNAME_SUMMARY':'No additional information for this item.', 'ENNAME_GROUP': 'Others'},
    '3': {'JPNAME_SUMMARY':'この項目に関する追加情報はありません。', 'JPNAME_GROUP': 'その他'},
    '4': {'SCNAME_SUMMARY':'本项无补充说明。', 'SCNAME_GROUP': '其他'}
}

# 保留的欄位清單（用於去重）
SUBSET_2_KEEP = ['ITEM_CODE', 'RECORD_ID', 'LANG_NO', 'ORG_ID',
                 'TCNAME_SUMMARY', 'ENNAME_SUMMARY', 'JPNAME_SUMMARY', 'SCNAME_SUMMARY']


# 讀取主表，移除重複的row
def get_unique_rows(final_df: pd.DataFrame) -> pd.DataFrame:
    # 缺值補空字串
    df_all = final_df.fillna('')

    # df_all 中，依照 SUBSET_2_KEEP 指定的欄位組合，移除重複 row
    df_unique = df_all.drop_duplicates(subset=SUBSET_2_KEEP, keep='first')

    return df_unique


# 整理 COMMENT、SUMMARY、GROUP、ITEM 欄位內的空行與空值
def postprocess_multilang(final_df: pd.DataFrame) -> pd.DataFrame:

    # COMMENT 移除換行、空行符號、轉為 str
    final_df['COMMENT'] = (
        final_df['COMMENT']
        .fillna('')
        .str.replace(r'\r|\n', '', regex=True)
        .str.translate(str.maketrans({
            '（': '(', '）': ')',
            '【': '[', '】': ']',
            '：': ':', '；': ';',
            '，': ',', '。': '.',
            '！': '!', '？': '?',
            '“': '"', '”': '"',
            '‘': "'", '’': "'",
            '、': ',', '　': ' ',
            '～': '~', '％': '%', '＋': '+', '－': '-', '＝': '=', '＠': '@'
        }))
        .str.replace(r'\s*\(\s*', '(', regex=True)
        .str.replace(r'\s*\)\s*', ')', regex=True)
        .astype(str)
    )

    # SUMMARY 移除換行、空行符號、空值填入 LANGU_DEFAULT_MAP 預設值
    for i, summary in enumerate(['TCNAME_SUMMARY', 'ENNAME_SUMMARY', 'JPNAME_SUMMARY', 'SCNAME_SUMMARY']):
        final_df[summary] = final_df[summary].fillna('').str.replace(r'\r|\n', '', regex=True)
        final_df[summary] = final_df[summary].replace('', LANGU_DEFAULT_MAP[str(i+1)][summary])

    # GROUPNO '其他' 更改排序於最後顯示
    max_groupno = final_df['GROUPNO'].max()
    final_df['GROUPNO'] = final_df['GROUPNO'].replace(0, max_groupno + 1)
    final_df['GROUPNO'] = final_df['GROUPNO'].fillna(max_groupno + 1).astype(int)

    # GROUP 移除換行、空行符號、空值填入 LANGU_DEFAULT_MAP 預設值
    for i, group in enumerate(['TCNAME_GROUP', 'ENNAME_GROUP', 'JPNAME_GROUP', 'SCNAME_GROUP']):
        final_df[group] = final_df[group].fillna('').str.replace(r'\r|\n', '', regex=True)
        final_df[group] = final_df[group].replace('', LANGU_DEFAULT_MAP[str(i+1)][group])

    df_unique = get_unique_rows(final_df)

    # 依照 RECORD_ID、GROUPNO、TCNAME_ITEM 排序
    df_unique.sort_values(by=['RECORD_ID', 'GROUPNO', 'TCNAME_ITEM'], inplace=True, kind='mergesort')

    return df_unique
