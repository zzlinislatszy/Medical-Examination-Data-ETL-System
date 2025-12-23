import os
import time
import re
import logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

LANGU_DEFAULT_TEXT = ['本項無補充說明', 'No additional information for this item.', 'この項目に関する追加情報はありません。', '本项无补充说明。']

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SuggestionTranslator:

    SYSTEM_PROMPT = {
        # TC
        '1': {'system_prompt':
        ("你是一位專業報告文字編輯，擅長將專業術語改寫成易懂、口吻中性且不過度承諾的白話文。\n"
        "請遵守以下原則：\n"
        "1) 繁體中文，不要簡體字。\n"
        "2) 不臆測、不新增原文未提及的資訊。\n"
        "3) 保留數字與時間單位(例：3個月、每週)。\n"
        "4) 將艱澀術語改為一般人能懂的說法。\n"
        "5) 口吻中性、尊重、具可執行性。\n"
        "6) 僅輸出『改寫後的單一段文字』，不要加標題或前綴。\n"
        "7) 單段落1~3句，盡量不超過60字，總長度不超過300字。\n"
        "8) 譯文以敘述句呈現，不要加入「如果...那麼...」等語氣開頭\n"),
        'user_prompt':
        "請將以下內容改寫為專業且易讀、好理解的文字，並且結構及語言要與原文接近："},

        # EN
        '2':{'system_prompt':
        ("You are a professional report editor skilled at rewriting domain terms into plain, neutral language without overpromising.\n"
        "Please follow these principles:\n"
        "1) Do not speculate or add any information not mentioned in the original text.\n"
        "2) Keep all numbers and time units.\n"
        "3) Replace difficult terms with everyday language understandable to the general public.\n"
        "4) Maintain a neutral, respectful, and actionable tone.\n"
        "5) Output only the rewritten single paragraph — do not include any titles or prefixes.\n"
        "6) Write 1–3 sentences per paragraph, with a total length not exceeding 300 characters.\n"
        "7) Use declarative sentences only; avoid starting with conditional phrases like 'If... then...'.\n"),
        'user_prompt':
        'Please rewrite the following text into professional, readable, and easy-to-understand language, while keeping the structure and tone close to the original text:'},

        # JP
        '3':{'system_prompt':
        ("あなたは専門レポートのライターであり、専門用語をわかりやすく、中立的で誇張のない口調に書き換えることが得意です。\n"
        "次の原則に従ってください：\n"
        "1) 原文に記載されていない情報を推測したり、追加したりしないこと。\n"
        "2) 数値や時間の単位は必ず残すこと。\n"
        "3) 難解な専門用語は一般の人が理解できる表現に置き換えること。\n"
        "4) 口調は中立的で、敬意を持ち、実行可能な内容にすること。\n"
        "5) 出力は改写後の単一の段落のみとし、タイトルや前置きは加えないこと。\n"
        "6) 段落は1〜3文、全体で300字を超えないようにすること。\n"
        "7) 叙述文で書くこと。\n"),
        'user_prompt':
        "次の内容を、読みやすく理解しやすい表現に書き換えてください。文章の構成と言葉の調子は原文に近づけてください。"},

        # SC
        '4':{'system_prompt':
        ("你是一位专业报告文字编辑，擅长将专业术语改写为通俗易懂、语气中立且不过度承诺的文字。\n"
        "请遵守以下原则：\n"
        "1) 不臆测、不添加原文未提及的信息。"
        "2) 保留数字与时间单位。"
        "3) 将艰涩的术语改写为大众能理解的表达方式。"
        "4) 保持语气中立、尊重且具可执行性。"
        "5) 仅输出改写后的单一段文字，不要加标题或前缀。"
        "6) 简体中文。"
        "7) 每段1至3句，总长度不超过300字。"
        "8) 使用陈述句表达，不要以“如果……那么……”等语气开头。"),
        'user_prompt':
        "请将以下内容改写为专业、易读且容易理解的文字，并保持与原文相近的结构和语气："}
    }

    def __init__(self, langu_no: str, mode: str = 'azure', model: str = 'gpt-4o', max_workers: int = 3):
        """
        : param mode: 'azure'（此 demo 僅保留 azure 模式介面）
        : param model: Azure 部署名稱（由環境變數決定實際連線）
        : param max_workers: 並行處理數量
        """
        self.langu_no = langu_no
        self.mode = mode.lower()
        self.max_workers = max_workers
        self.max_retries = 3
        self.base_delay = 1

        if self.mode == 'azure':
            self._init_azure(model)
        else:
            raise ValueError(f"不支援的模式: {mode}，請使用'azure'")

    # 初始化 Azure OpenAI
    def _init_azure(self, deployment_name: str):
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')

        # 若未提供金鑰，改用 mock client（可離線跑 demo）
        if not endpoint or not api_key:
            self.client = None
            self.model = deployment_name
            logger.info("未設定 AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY，改用 mock 介面")
            return

        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{endpoint}/openai/deployments/{deployment_name}",
            default_query={'api-version': os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')},
            default_headers={'api-key': api_key},
        )
        self.model = deployment_name
        logger.info(f"使用 Azure OpenAI - 部署: {deployment_name}")

    def translate_batch(self, suggestions: List[str]) -> Dict[str, str]:
        """
        批次改寫多筆文本
        : param suggestions: texts
        : returns: key -> 原文，value -> 改寫後文本
        """
        if not suggestions:
            logger.warning("清單為空")
            return {}

        logger.info(f"開始處理 {len(suggestions)} 筆文本")
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for suggestion in suggestions:
                if suggestion in LANGU_DEFAULT_TEXT:
                    results[suggestion] = suggestion
                    continue

                future = executor.submit(self._translate_single, suggestion)
                futures[future] = suggestion

            for future in as_completed(futures):
                suggestion = futures[future]
                try:
                    results[suggestion] = future.result()
                except Exception as e:
                    logger.error(f"處理失敗 - {suggestion[:50]}...: {e}")
                    results[suggestion] = suggestion

        logger.info(f"完成 {len(results)} 筆")
        return results

    def _translate_single(self, suggestion: str) -> str:
        """
        逐一改寫
        """
        # mock：未設定 client 時，直接回傳包裝後文字（保留介面與流程）
        if self.client is None:
            return f"[LLM_OUTPUT]{suggestion}"

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': self.SYSTEM_PROMPT[self.langu_no]['system_prompt']},
                        {'role': 'user', 'content': f"{self.SYSTEM_PROMPT[self.langu_no]['user_prompt']}{suggestion}"}
                    ],
                    max_tokens=300,
                    temperature=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    top_p=1,
                )

                translated = response.choices[0].message.content.strip()
                logger.info(f"成功: {suggestion[:30]}...")
                return translated

            except Exception as e:
                if self._is_rate_limit_error(e):
                    wait_time = self._get_retry_wait_time(str(e), attempt)
                    logger.warning(f"達到速率限制，等待 {wait_time:.1f}秒 (第{attempt+1}/{self.max_retries}次)")
                    time.sleep(wait_time)

                    if attempt == self.max_retries - 1:
                        logger.error(f"達到最大重試次數 - {suggestion[:50]}...")
                        return suggestion
                else:
                    logger.error(f"處理錯誤 - {suggestion[:50]}...: {e}")
                    return suggestion

        return suggestion

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        error_str = str(error).lower()
        return 'rate_limit' in error_str or '429' in error_str

    def _get_retry_wait_time(self, error_msg: str, attempt: int) -> float:
        match = re.search(r'try again in ([\d.]+)s', error_msg)
        if match:
            return float(match.group(1)) + 0.1

        return self.base_delay * (2 ** attempt)


def process_suggestion(langu_no:str, suggestion_list: List[str], mode: str = 'azure', model: str = 'gpt-4o') -> Dict[str, str]:
    """
    : param suggestion_list: 文本列表
    : param mode: 'azure'
    : param model: 部署名稱
    : return: key -> 原文，value -> 改寫後文本
    """
    translator = SuggestionTranslator(langu_no=langu_no, mode=mode, model=model)
    return translator.translate_batch(suggestion_list)
