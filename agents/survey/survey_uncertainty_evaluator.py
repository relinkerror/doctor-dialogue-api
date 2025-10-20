from utils.embedding_similarity import EmbeddingSimilarityTool
import json
import re

class SurveyUncertaintyEvaluator:
    """
    每轮评估传入完整问卷所有题（含id/question/depends_on/status），prob_cache用问题文本或id为key均可。
    """

    def __init__(self, llm_client, eval_prompt, threshold=0.1, embedder_model="models/paraphrase-multilingual-MiniLM-L12-v2"):
        self.llm_client = llm_client
        self.eval_prompt = eval_prompt
        self.threshold = threshold
        self.sim_tool = EmbeddingSimilarityTool(embedder_model)
        self.prob_cache = {}
        self._prev_eval_ctx = ""      # 上一次评估时的上下文拼接文本
        self._last_n_for_ctx = 4      # 每次评估时取最近 n 轮对话构建上下文（可调）


    def combine_uncertainty(self, prev, new, sim):
        base = max(max(prev, new), 1e-6)
        return min((prev * new) / (base ** sim), 1.0)

    def _merge_ctx(self, dialog_history, last_n):
        if len(dialog_history) < last_n * 2:
            start = 0
        else:
            start = -last_n * 2
        return " ".join([d["content"] for d in dialog_history[start:]])
    
    def _build_eval_ctx(self, dialog_history, last_n=None):
        """把最近 last_n 轮（user+assistant 共2*last_n条）拼成一个字符串"""
        if last_n is None:
            last_n = self._last_n_for_ctx
        if len(dialog_history) < last_n * 2:
            start = 0
        else:
            start = -last_n * 2
        return " ".join([d["content"] for d in dialog_history[start:]])


    def evaluate(self, questions, dialog_history):
        if not questions:
            return []

        # 与上次评估上下文的相似度
        curr_ctx = self._build_eval_ctx(dialog_history, self._last_n_for_ctx)
        prev_ctx = self._prev_eval_ctx or ""
        sim = self.sim_tool.similarity(curr_ctx, prev_ctx) if prev_ctx else 1.0
        print(f"[EVAL][UE] q_count={len(questions)} threshold={self.threshold} ctx_sim_prev={sim:.3f}")

        payload = {'items': questions}
        messages = [
            {'role': 'system', 'content': self.eval_prompt},
            {'role': 'system', 'content': json.dumps(payload, ensure_ascii=False)}
        ] + dialog_history

        raw = self.llm_client.call_json(messages)
        # 兼容 dict / str
        if isinstance(raw, dict):
            parsed = raw
        else:
            s = str(raw)
            m = re.search(r"(\{[\s\S]*\})", s)
            parsed = json.loads(m.group(1)) if m else {}
        results = parsed.get('uncertainties', [])
        print(f"[EVAL][UE] model_returns={len(results)}")

        updates = []
        for res in results:
            qid = res.get("id")
            question = res.get("question")
            pt = float(res.get('PT', res.get('prob_true', 0.0)))
            pf = float(res.get('PF', res.get('prob_false', 0.0)))
            key = qid or question

            prev_true  = self.prob_cache.get(key, {}).get("uncertainty_true", 1.0)
            prev_false = self.prob_cache.get(key, {}).get("uncertainty_false", 1.0)
            u_t, u_f = 1 - pt, 1 - pf
            fu_t = self.combine_uncertainty(prev_true,  u_t, sim)
            fu_f = self.combine_uncertainty(prev_false, u_f, sim)
            self.prob_cache[key] = {"uncertainty_true": fu_t, "uncertainty_false": fu_f}
            print(f"[EVAL][UE][{key}] PT={pt:.3f} PF={pf:.3f} U_T(prev->{fu_t:.3f}) U_F(prev->{fu_f:.3f})")

            status = None
            if fu_t < self.threshold and fu_f < self.threshold:
                status = True if fu_t <= fu_f else False
            elif fu_t < self.threshold:
                status = True
            elif fu_f < self.threshold:
                status = False
            if status is not None:
                updates.append({"id": qid, "question": question, "status": status})

        self._prev_eval_ctx = curr_ctx
        print(f"[EVAL][UE] updates_count={len(updates)}")
        return updates
