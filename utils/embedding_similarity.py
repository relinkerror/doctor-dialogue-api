# utils/embedding_similarity.py
import os
import hashlib
import pickle
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

# ======= 全局模型单例缓存 =======
_EMBEDDER_SINGLETONS: Dict[str, SentenceTransformer] = {}

# 环境变量（可选覆盖）
_DEFAULT_MODEL_ID = os.getenv(
    "EMBEDDER_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
_HF_CACHE_DIR = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", "/opt/render/.cache/huggingface"))

def _has_weights(dir_path: str) -> bool:
    """目录存在且包含常见权重文件即视为可用本地模型。"""
    if not os.path.isdir(dir_path):
        return False
    candidates = (
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
        os.path.join("onnx", "model.onnx"),
    )
    return any(os.path.exists(os.path.join(dir_path, f)) for f in candidates)

def _choose_load_target(model_path_arg: Optional[str]) -> str:
    """
    决策顺序（对外兼容且更健壮）：
    1) 如果入参是本地目录且有权重 → 用本地目录
    2) 否则若设置了 ENV: EMBEDDER_MODEL → 用该 HF 模型 ID
    3) 否则若入参看起来像 HF 模型ID（含 '/'）→ 用入参
    4) 否则回退到默认 HF 模型 ID（sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2）
    """
    arg = (model_path_arg or "").strip()

    # 1) 本地目录且有权重
    if arg and os.path.isdir(arg) and _has_weights(arg):
        print(f"[EmbeddingCache] Using LOCAL directory with weights: {arg}")
        return arg

    # 2) 环境变量优先
    if _DEFAULT_MODEL_ID:
        if arg and os.path.isdir(arg) and not _has_weights(arg):
            print(f"[EmbeddingCache] WARNING: Local dir '{arg}' has NO weights; "
                  f"fallback to HF model id from ENV: {_DEFAULT_MODEL_ID}")
        elif arg and not os.path.isdir(arg):
            print(f"[EmbeddingCache] '{arg}' is not a valid local dir; "
                  f"fallback to HF model id from ENV: {_DEFAULT_MODEL_ID}")
        else:
            print(f"[EmbeddingCache] Using HF model id from ENV: {_DEFAULT_MODEL_ID}")
        return _DEFAULT_MODEL_ID

    # 3) 入参像 HF 模型ID
    if "/" in arg:
        print(f"[EmbeddingCache] Using HF model id from argument: {arg}")
        return arg

    # 4) 兜底默认
    print("[EmbeddingCache] Fallback to default HF model id: "
          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class EmbeddingSimilarityTool:
    """
    向后兼容 + 更健壮：
    - 仍接受你原来的 model_path 形参（默认 'models/paraphrase-multilingual-MiniLM-L12-v2'）
    - 若该目录不存在或无权重，自动回退到 EMBEDDER_MODEL（ENV）或默认 HF 模型ID
    - 缓存目录使用 HF_HOME/TRANSFORMERS_CACHE，默认 /opt/render/.cache/huggingface
    - 类/方法签名不变，不影响 SurveyAgent/前端调用
    """
    def __init__(
        self,
        model_path: str = "models/paraphrase-multilingual-MiniLM-L12-v2",
        cache_path: str = "./utils/cache/embedding_cache.pkl",
        save_every: int = 10
    ):
        global _EMBEDDER_SINGLETONS

        # 确保缓存目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if _HF_CACHE_DIR:
            os.makedirs(_HF_CACHE_DIR, exist_ok=True)

        # 关键：选择加载目标（本地可用 → 本地；否则强制回退到 HF）
        load_target = _choose_load_target(model_path)

        key = f"{load_target}::cache={_HF_CACHE_DIR or 'none'}"
        if key not in _EMBEDDER_SINGLETONS:
            print(f"[EmbeddingCache] Loading SentenceTransformer: {load_target}")
            if _HF_CACHE_DIR:
                _EMBEDDER_SINGLETONS[key] = SentenceTransformer(load_target, cache_folder=_HF_CACHE_DIR)
            else:
                _EMBEDDER_SINGLETONS[key] = SentenceTransformer(load_target)
        else:
            print(f"[EmbeddingCache] Reusing loaded SentenceTransformer: {load_target}")

        self.embedder = _EMBEDDER_SINGLETONS[key]
        self.cache_path = cache_path
        self.save_every = save_every
        self._save_count = 0

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.persist_cache = pickle.load(f)
                print(f"[EmbeddingCache] Loaded {len(self.persist_cache)} question embeddings from {cache_path}")
            except Exception as e:
                print(f"[EmbeddingCache] WARNING: failed to load cache file '{cache_path}': {e}")
                self.persist_cache = {}
        else:
            self.persist_cache = {}

        self.memory_cache: Dict[str, np.ndarray] = {}

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_embedding(self, text: str, persist: bool = False) -> np.ndarray:
        if not text:
            return np.zeros((384,), dtype=np.float32)  # 兼容空文本，避免异常
        key = self._text_hash(text)

        # 内存命中
        if key in self.memory_cache:
            return self.memory_cache[key]

        # 持久化命中（通常只对标准题目持久化）
        if persist and key in self.persist_cache:
            self.memory_cache[key] = self.persist_cache[key]
            return self.persist_cache[key]

        # 重新计算（normalize_embeddings=True）
        emb = self.embedder.encode([text], normalize_embeddings=True)[0]
        self.memory_cache[key] = emb

        if persist:
            self.persist_cache[key] = emb
            self._save_count += 1
            if self._save_count % self.save_every == 0:
                self.save()

        return emb

    def similarity(self, s1: str, s2: str, persist_q1: bool = False, persist_q2: bool = False) -> float:
        """
        persist_q1 / persist_q2 控制是否写入持久化cache（一般 s1=用户输入不持久化，s2=标准题目持久化）
        """
        if not s1 or not s2:
            return 0.0
        v1 = self._get_embedding(s1, persist=persist_q1)
        v2 = self._get_embedding(s2, persist=persist_q2)
        return float(np.dot(v1, v2))

    def save(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.persist_cache, f)
            print(f"[EmbeddingCache] Saved {len(self.persist_cache)} question embeddings to {self.cache_path}")
        except Exception as e:
            print(f"[EmbeddingCache] WARNING: failed to save cache file '{self.cache_path}': {e}")

if __name__ == "__main__":
    tool = EmbeddingSimilarityTool()
    ctx = "我最近晚上总是睡不好，有点抑郁"
    questions = [
        "两周内是否持续情绪低落？",
        "最近晚上总是睡不好，有些抑郁的感觉吗？",
        "最近是否有自杀的想法？",
        "经常早醒或睡眠质量很差吗？",
        "最近觉得对什么都不感兴趣，做什么都提不起劲？",
    ]
    print("\n第一次：embedding cache 全空，只持久化标准问题。")
    for q in questions:
        sim = tool.similarity(ctx, q, persist_q1=False, persist_q2=True)
        print(f"相似度: {sim:.3f} | 问题: {q}")

    print("\n第二次：embedding 应命中 cache，速度更快。")
    for q in questions:
        sim = tool.similarity(ctx, q, persist_q1=False, persist_q2=True)
        print(f"相似度: {sim:.3f} | 问题: {q}")

    tool.save()
