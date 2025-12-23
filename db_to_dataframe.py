from utils import log_execution_time
from typing import List, Dict, Any
import pandas as pd
import pymongo
import os

SUBSET = [
    'RECORD_ID', 'ORG_ID', 'LANG_NO', 'DIAG_CODE',
    'GROUPNO', 'TCNAME_GROUP', 'ENNAME_GROUP', 'JPNAME_GROUP', 'SCNAME_GROUP',
    'ITEM_CODE', 'TCNAME_ITEM', 'ENNAME_ITEM', 'JPNAME_ITEM', 'SCNAME_ITEM',
    'COMMENT', 'ENNAME_COMMENT', 'JPNAME_COMMENT', 'SCNAME_COMMENT',
    'TCNAME_SUMMARY', 'ENNAME_SUMMARY', 'JPNAME_SUMMARY', 'SCNAME_SUMMARY'
]


@log_execution_time
def db_to_dataframe(api_request: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Demo 版：將 API 輸入的 JSON 資料轉換為 DataFrame，並以 pymongo 示範擴充查詢（連線資訊以環境變數提供）。

    前提條件：
    1) MONGODB_URI / DB 與 collection 名稱需由環境變數提供（避免在程式碼中出現任何內部資訊）
    2) 若未提供環境變數，會使用「demo fallback」資料，確保範例可離線執行

    Args:
        api_request: 包含多個 record 的列表，每個 record 包含 items 和 findings 資訊。

    Returns:
        pd.DataFrame: 擴充後的資料框（示範用欄位）。
    """

    # 1 展開 input（保留原本 json_normalize + explode 的管線風格）
    df_raw = pd.json_normalize(api_request, record_path='ITEMS', meta=['RECORD_ID', 'LANG_NO', 'ORG_ID'])
    df_exploded = df_raw.explode('FINDINGS').reset_index(drop=True)
    df_base = pd.concat([
        df_exploded.drop('FINDINGS', axis=1),
        df_exploded['FINDINGS'].apply(pd.Series)
    ], axis=1)

    # drop COMMENT 為 "" 的 row
    df_base['COMMENT_clean'] = df_base['COMMENT'].fillna('').astype(str).str.strip()
    rows_to_drop = (df_base['COMMENT_clean'] == '')
    df_base = df_base[~rows_to_drop].copy()
    df_base = df_base.drop(columns=['COMMENT_clean'])

    # 2 連 MongoDB（連線/庫/表名稱一律由環境變數提供）
    mongo_uri = os.getenv('MONGODB_URI', '')
    main_db_name = os.getenv('MONGODB_DB_MAIN', '')
    aux_db_name = os.getenv('MONGODB_DB_AUX', '')

    col_item_meta = os.getenv('MONGODB_COL_ITEM_META', '')
    col_item_group_map = os.getenv('MONGODB_COL_ITEM_GROUP_MAP', '')
    col_diag = os.getenv('MONGODB_COL_DIAG', '')
    col_summary = os.getenv('MONGODB_COL_SUMMARY', '')

    # 若缺少任何一項設定，就改用 fallback（避免硬編任何內部資訊）
    use_fallback = not all([mongo_uri, main_db_name, aux_db_name, col_item_meta, col_item_group_map, col_diag, col_summary])

    unique_items_list = df_base.ITEM_CODE.astype(str).str.strip().unique().tolist()

    if use_fallback:
        # ---- Demo fallback datasets ----
        item_meta = pd.DataFrame([{
            'ITEM_CODE': code,
            'TCNAME_ITEM': f'項目 {code}',
            'SCNAME_ITEM': f'项目 {code}',
            'JPNAME_ITEM': f'項目 {code}',
            'ENNAME_ITEM': f'Item {code}',
            'ORG_ID': str(df_base.ORG_ID.iloc[0]).strip()
        } for code in unique_items_list])

        item_group_map = pd.DataFrame([{
            'ITEM_CODE': code,
            'GROUPNO': 1,
            'TCNAME_GROUP': '範例分類',
            'ENNAME_GROUP': 'Sample Group',
            'JPNAME_GROUP': 'サンプル分類',
            'SCNAME_GROUP': '示例分类'
        } for code in unique_items_list])

        diag_tbl = pd.DataFrame([{
            'DIAG_CODE': str(code).strip(),
            'SUMMARY_CODE': str(code).strip(),
            'ENNAME_COMMENT': '',
            'JPNAME_COMMENT': '',
            'SCNAME_COMMENT': ''
        } for code in df_base.DIAG_CODE.astype(str).str.strip().unique().tolist()])

        summary_tbl = pd.DataFrame([{
            'SUMMARY_CODE': str(code).strip(),
            'TCNAME_SUMMARY': '',
            'SCNAME_SUMMARY': '',
            'ENNAME_SUMMARY': '',
            'JPNAME_SUMMARY': ''
        } for code in diag_tbl.SUMMARY_CODE.unique().tolist()])

    else:
        client = pymongo.MongoClient(mongo_uri)
        DB_MAIN = client[main_db_name]
        DB_AUX = client[aux_db_name]

        # FOR: 查 ITEM_NAME（多語系顯示名稱）
        item_meta_cur = DB_MAIN[col_item_meta].find(
            {"ITEM_CODE": {"$in": unique_items_list}},
            {"ITEM_CODE": 1, "TCNAME": 1, "SCNAME": 1, "JPNAME": 1, "ENNAME": 1, "ORG_ID": 1, "_id": 0}
        )
        item_meta = pd.DataFrame(list(item_meta_cur))
        item_meta.rename(columns={'TCNAME': 'TCNAME_ITEM',
                                  'JPNAME': 'JPNAME_ITEM',
                                  'ENNAME': 'ENNAME_ITEM',
                                  'SCNAME': 'SCNAME_ITEM'}, inplace=True)

        # FOR: 查 GROUPNO、GROUP_NAME
        item_group_map_cur = DB_AUX[col_item_group_map].find(
            {"ITEM_CODE": {"$in": unique_items_list}},
            {"_id": 0}
        )
        item_group_map = pd.DataFrame(list(item_group_map_cur))

        # FOR: 查 SUMMARY_CODE
        diag_cur = DB_MAIN[col_diag].find(
            {},
            {"DIAG_CODE": 1, "SUMMARY_CODE": 1,
             "SCNAME": 1, "ENNAME": 1, "JPNAME": 1,
             "ORG_ID": 1, "_id": 0}
        )
        diag_tbl = pd.DataFrame(list(diag_cur))
        diag_tbl.rename(columns={'JPNAME': 'JPNAME_COMMENT',
                                 'ENNAME': 'ENNAME_COMMENT',
                                 'SCNAME': 'SCNAME_COMMENT'}, inplace=True)

        # FOR: 查 SUMMARY_NAME
        summary_cur = DB_AUX[col_summary].find(
            {},
            {"SUMMARY_CODE": 1, "TCNAME": 1, "SCNAME": 1, "JPNAME": 1, "ENNAME": 1, "ORG_ID": 1, "_id": 0}
        )
        summary_tbl = pd.DataFrame(list(summary_cur))
        summary_tbl.rename(columns={'TCNAME': 'TCNAME_SUMMARY',
                                    'JPNAME': 'JPNAME_SUMMARY',
                                    'ENNAME': 'ENNAME_SUMMARY',
                                    'SCNAME': 'SCNAME_SUMMARY'}, inplace=True)

    # 3 PreProcessing & Merge（保留 merge 流程）
    df_base['ITEM_CODE'] = df_base['ITEM_CODE'].astype(str).str.strip()
    df_base['ORG_ID'] = df_base['ORG_ID'].astype(str).str.strip()
    df_base['DIAG_CODE'] = df_base['DIAG_CODE'].astype(str).str.strip()

    item_meta['ITEM_CODE'] = item_meta['ITEM_CODE'].astype(str).str.strip()
    item_meta['ORG_ID'] = item_meta['ORG_ID'].astype(str).str.strip()

    item_group_map['ITEM_CODE'] = item_group_map['ITEM_CODE'].astype(str).str.strip()

    diag_tbl['DIAG_CODE'] = diag_tbl['DIAG_CODE'].astype(str).str.strip()
    if 'ORG_ID' in diag_tbl.columns:
        diag_tbl['ORG_ID'] = diag_tbl['ORG_ID'].astype(str).str.strip()
    if 'SUMMARY_CODE' in diag_tbl.columns:
        diag_tbl['SUMMARY_CODE'] = diag_tbl['SUMMARY_CODE'].astype(str).str.strip()

    if 'SUMMARY_CODE' in summary_tbl.columns:
        summary_tbl['SUMMARY_CODE'] = summary_tbl['SUMMARY_CODE'].astype(str).str.strip()
    if 'ORG_ID' in summary_tbl.columns:
        summary_tbl['ORG_ID'] = summary_tbl['ORG_ID'].astype(str).str.strip()

    merged_item_name = df_base.merge(item_meta, on=['ITEM_CODE', 'ORG_ID'], how='left')
    merged_group = merged_item_name.merge(item_group_map, on='ITEM_CODE', how='left', suffixes=('_ITEM', '_GROUP'))

    # DIAG -> SUMMARY_CODE
    if 'ORG_ID' in diag_tbl.columns:
        merged_for_summary = merged_group.merge(diag_tbl, on=['DIAG_CODE'], how='left')
    else:
        merged_for_summary = merged_group.merge(diag_tbl, on=['DIAG_CODE'], how='left')

    # SUMMARY_CODE -> SUMMARY_NAME
    if 'ORG_ID' in summary_tbl.columns:
        final_df = merged_for_summary.merge(summary_tbl, on=['SUMMARY_CODE'], how='left')
    else:
        final_df = merged_for_summary.merge(summary_tbl, on=['SUMMARY_CODE'], how='left')

    # 保障必要欄位存在（避免 demo fallback 或實際表欄位略有不同）
    for col in SUBSET:
        if col not in final_df.columns:
            final_df[col] = ''

    return final_df[SUBSET]
