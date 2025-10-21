# utils/embedding_similarity.py
import os
import hashlib
import pickle
from typing import Dict
from sentence_transformers import SentenceTransformer
import numpy as np

# ======= 全局模型单例缓存 =======
_EMBEDDER_SINGLETONS: Dict[str, SentenceTransformer] = {}

# 允许通过环境变量覆盖模型ID与缓存目录
_DEFAULT_MODEL_ID = os.getenv(
    "EMBEDDER_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
_HF_CACHE_DIR = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", "/opt/render/.cache/huggingface"))

def _looks_like_local_model_dir(p: str) -> bool:
    """存在且包含常见权重文件即视为本地模型目录"""
    if not os.path.isdir(p):
        return False
    candidates = (
        "model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack",
        os.path.join("onnx", "model.onnx"),
    )
    return any(os.path.exists(os.path.join(p, f)) for f in candidates)

class EmbeddingSimilarityTool:
    """
    向后兼容的增强版：
    - 若传入本地目录存在权重 → 直接本地加载
    - 若目录不存在或缺权重 → 将参数视为模型ID；若也不像ID，则退回默认ID
    - 缓存目录使用 HF_HOME/TRANSFORMERS_CACHE，默认 /opt/render/.cache/huggingface
    - 类/方法签名不变，不影响你的 SurveyAgent/前端调用
    """
    def __init__(
        self,
        model_path: str = "models/paraphrase-multilingual-MiniLM-L12-v2",
        cache_path: str = "./utils/cache/embedding_cache.pkl",
        save_every: int = 10
    ):
        global _EMBEDDER_SINGLETONS

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if _HF_CACHE_DIR:
            os.makedirs(_HF_CACHE_DIR, exist_ok=True)

        # 决定加载目标
        load_target = model_path
        if _looks_like_local_model_dir(model_path):
            # 本地目录且有权重 → 用本地
            pass
        else:
            # 不像本地，就按模型ID处理；若也不像ID（不含 "/"），退回默认ID
            if "/" not in model_path or not model_path.strip():
                load_target = _DEFAULT_MODEL_ID

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
            with open(cache_path, "rb") as f:
                self.persist_cache = pickle.load(f)
            print(f"[EmbeddingCache] Loaded {len(self.persist_cache)} question embeddings from {cache_path}")
        else:
            self.persist_cache = {}
        self.memory_cache: Dict[str, np.ndarray] = {}

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_embedding(self, text: str, persist: bool = False) -> np.ndarray:
        key = self._text_hash(text)
        # 先查内存
        if key in self.memory_cache:
            return self.memory_cache[key]
        # 再查持久化（仅用于标准问题）
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
        persist_q1, persist_q2 控制是否写入持久化cache（默认仅问题端持久化）
        一般用法：用户输入 s1 不持久化，题目 s2 持久化
        """
        if not s1 or not s2:
            return 0.0
        v1 = self._get_embedding(s1, persist=persist_q1)
        v2 = self._get_embedding(s2, persist=persist_q2)
        return float(np.dot(v1, v2))

    def save(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.persist_cache, f)
        print(f"[EmbeddingCache] Saved {len(self.persist_cache)} question embeddings to {self.cache_path}")

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
    print("\n第一次：embedding cache全空，只持久化标准问题。")
    for q in questions:
        sim = tool.similarity(ctx, q, persist_q1=False, persist_q2=True)
        print(f"相似度: {sim:.3f} | 问题: {q}")

    print("\n第二次：embedding应全命中cache，速度极快。")
    for q in questions:
        sim = tool.similarity(ctx, q, persist_q1=False, persist_q2=True)
        print(f"相似度: {sim:.3f} | 问题: {q}")

    tool.save()
