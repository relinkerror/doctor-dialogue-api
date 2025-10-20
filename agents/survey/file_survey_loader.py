import os
import json
import re
import hashlib
from typing import Dict, List, Any
from utils.LLMClient import LLMClient

def file_content_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(4096):
            h.update(chunk)
    return h.hexdigest()

def survey_cache_key(meta_json_path, questionnaire_dir):
    # meta内容hash
    meta_hash = file_content_hash(meta_json_path)
    # 问卷目录下所有文件hash，合并再hash
    all_files = sorted([
        os.path.join(questionnaire_dir, fn)
        for fn in os.listdir(questionnaire_dir)
        if os.path.isfile(os.path.join(questionnaire_dir, fn))
    ])
    content_bytes = b''
    for fn in all_files:
        with open(fn, "rb") as f:
            content_bytes += f.read()
    survey_hash = hashlib.md5(content_bytes).hexdigest()
    return f"{meta_hash}_{survey_hash}"

def get_cache_path(meta_json_path, questionnaire_dir, cache_dir="./depends_on_cache"):
    key = survey_cache_key(meta_json_path, questionnaire_dir)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"depends_on_{key}.json")

def load_depends_on_cache(meta_json_path, questionnaire_dir, cache_dir="./depends_on_cache"):
    path = get_cache_path(meta_json_path, questionnaire_dir, cache_dir)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def cache_depends_on(meta_json_path, questionnaire_dir, surveys, cache_dir="./depends_on_cache"):
    path = get_cache_path(meta_json_path, questionnaire_dir, cache_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(surveys, f, ensure_ascii=False, indent=2)

class FileSurveyLoader:
    """
    工程化 Loader，支持通过 meta json 加载所有问卷原文，
    自动调用 LLM 补齐每题 depends_on 字段，并缓存本地，按问卷内容唯一。
    """

    def __init__(self, meta_json_path: str, llm_client: LLMClient, survey_depends_on_prompt: str = None):
        self.meta_json_path = meta_json_path
        with open(meta_json_path, encoding='utf-8') as f:
            self.meta = json.load(f)
        self.llm_client = llm_client

        # 如果未传入prompt，则加载默认
        if not survey_depends_on_prompt or not survey_depends_on_prompt.strip():
            # 默认config下
            default_prompt_path = os.path.join(
                os.path.dirname(meta_json_path), "survey_depends_on_prompt.txt"
            )
            if os.path.exists(default_prompt_path):
                with open(default_prompt_path, encoding="utf-8") as f:
                    self.survey_depends_on_prompt = f.read()
                print(f"[INFO] 已自动加载默认depends_on prompt: {default_prompt_path}")
            else:
                raise ValueError("未提供survey_depends_on_prompt，且未找到默认配置文件。")
        else:
            self.survey_depends_on_prompt = survey_depends_on_prompt

    def load_surveys(self, questionnaire_dir: str, use_cache=True, cache_dir="./depends_on_cache") -> Dict[str, List[Dict[str, Any]]]:
        """
        主流程：加载问卷并自动补齐depends_on，结果按问卷内容缓存。
        Returns:
            Dict[survey_id, List[Dict]]（每题含id、question、status、depends_on、remarks）
        """
        # 1. 检查缓存
        if use_cache:
            cache = load_depends_on_cache(self.meta_json_path, questionnaire_dir, cache_dir)
            if cache is not None:
                print(f"[CACHE] 命中依赖补齐缓存: {get_cache_path(self.meta_json_path, questionnaire_dir, cache_dir)}")
                return cache

        surveys: Dict[str, List[Dict[str, Any]]] = {}
        # 2. 按 meta 加载所有原文问卷
        for entry in self.meta['surveys']:
            sid = entry['id']
            file_path = os.path.join(questionnaire_dir, entry['file'])
            items: List[Dict[str, Any]] = []
            if not os.path.isfile(file_path):
                print(f"[WARNING] 未找到问卷文件: {file_path}")
                continue
            with open(file_path, encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    text = line.strip()
                    if not text:
                        continue
                    qid = f"{sid}_Q{idx+1}"
                    items.append({
                        'id': qid,
                        'question': text,
                        'status': None,
                        'depends_on': [],
                        'remarks': []
                    })
            surveys[sid] = items

        # 3. 对每份问卷单独调用 LLM 批量补齐 depends_on
        for sid, items in surveys.items():
            payload = {'survey': [{'id': it['id'], 'question': it['question']} for it in items]}
            messages = [
                {'role': 'system', 'content': self.survey_depends_on_prompt},
                {'role': 'system', 'content': json.dumps(payload, ensure_ascii=False)}
            ]
            parsed = None
            try:
                resp = self.llm_client.call_json(messages)
                parsed = resp if isinstance(resp, dict) else json.loads(resp)
            except Exception as e:
                raw = self.llm_client.call(messages)
                m = re.search(r'```json\s*([\s\S]*?)```', raw)
                json_str = m.group(1) if m else None
                if not json_str:
                    m2 = re.search(r'(\{[\s\S]*\})', raw)
                    json_str = m2.group(1) if m2 else '{}'
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"[ERROR] depends_on JSON 解析失败 for {sid}:", raw)
                    parsed = {}
            dep_list = parsed.get('depends_on_map') or parsed.get('items') or []
            id_map = {entry['id']: entry.get('depends_on', []) for entry in dep_list}
            for it in items:
                it['depends_on'] = id_map.get(it['id'], [])
                print(f"[DEBUG][Update] {sid} - {it['id']} depends_on: {it['depends_on']}")

        # 4. 写入缓存
        if use_cache:
            cache_depends_on(self.meta_json_path, questionnaire_dir, surveys, cache_dir)
            print(f"[CACHE] 已缓存依赖补齐结果: {get_cache_path(self.meta_json_path, questionnaire_dir, cache_dir)}")
        return surveys

# ============= main 测试 =============

if __name__ == "__main__":
    # 1. 加载 LLM 配置
    llm_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/llm_config.json"))
    llm = LLMClient.from_config(llm_config_path)

    # 2. 加载 meta 和问卷原文目录
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/"))
    questionnaire_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../questionnaire/"))
    meta_json_path = os.path.join(config_dir, "dsm5_depression_surveys.json")

    # 3. 加载prompt文本
    prompt_path = os.path.join(config_dir, "survey_depends_on_prompt.txt")
    with open(prompt_path, encoding="utf-8") as f:
        survey_depends_on_prompt = f.read()

    # 4. 执行加载流程
    loader = FileSurveyLoader(meta_json_path, llm, survey_depends_on_prompt)
    surveys = loader.load_surveys(questionnaire_dir)

    # 5. 展示前几题依赖关系
    print("【Loader加载结果】:") 
    for sid, items in surveys.items():
        print(f"- Survey: {sid}, 共{len(items)}题，前3题:")
        for it in items[:3]:
            print(f"    {it['id']}: {it['question']}  depends_on: {it['depends_on']}")
