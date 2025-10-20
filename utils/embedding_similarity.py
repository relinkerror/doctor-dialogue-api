# embedding_similarity.py
import os
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# ======= 全局模型单例缓存 =======
_EMBEDDER_SINGLETONS = {}

class EmbeddingSimilarityTool:
    def __init__(self, model_path: str = "models/paraphrase-multilingual-MiniLM-L12-v2",
                 cache_path: str = "./utils/cache/embedding_cache.pkl", save_every=10):
        global _EMBEDDER_SINGLETONS
        if model_path not in _EMBEDDER_SINGLETONS:
            print(f"[EmbeddingCache] Loading SentenceTransformer: {model_path}")
            _EMBEDDER_SINGLETONS[model_path] = SentenceTransformer(model_path)
        else:
            print(f"[EmbeddingCache] Reusing loaded SentenceTransformer: {model_path}")
        self.embedder = _EMBEDDER_SINGLETONS[model_path]
        self.cache_path = cache_path
        self.save_every = save_every
        self._save_count = 0

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.persist_cache = pickle.load(f)
            print(f"[EmbeddingCache] Loaded {len(self.persist_cache)} question embeddings from {cache_path}")
        else:
            self.persist_cache = {}
        self.memory_cache = {}

    def _text_hash(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_embedding(self, text, persist=False):
        key = self._text_hash(text)
        # 先查内存
        if key in self.memory_cache:
            #print(f"[EmbeddingCache][MEM-HIT] '{text[:18]}...'")
            return self.memory_cache[key]
        # 再查持久化cache（只限于标准问题）
        if persist and key in self.persist_cache:
            #print(f"[EmbeddingCache][DISK-HIT] '{text[:18]}...'")
            self.memory_cache[key] = self.persist_cache[key]
            return self.persist_cache[key]
        # 重新计算
        #print(f"[EmbeddingCache][ENCODE] '{text[:18]}...'")
        emb = self.embedder.encode([text], normalize_embeddings=True)[0]
        self.memory_cache[key] = emb
        # 持久化仅对标准问题
        if persist:
            self.persist_cache[key] = emb
            self._save_count += 1
            if self._save_count % self.save_every == 0:
                self.save()
        return emb


    def similarity(self, s1: str, s2: str, persist_q1=False, persist_q2=False):
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
        "最近觉得对什么都不感兴趣，做什么都提不起劲？"
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
