class SurveySelector:
    """
    问卷最佳下一题选择器：对pending问题递归依赖链，自动选择最可问的问题。
    """
    def __init__(self, state_manager, similarity_tool, max_depth=10):
        self.state_manager = state_manager
        self.sim_tool = similarity_tool
        self.max_depth = max_depth

    def get_all_ready_leaves(self, dialog_history):
        """
        遍历所有pending question，递归展开依赖链到底，返回所有可推进叶子题和对应链
        """
        ctx_now = " ".join([d["content"] for d in dialog_history]) if dialog_history else ""
        leaves = []
        for q in self.state_manager.get_pending_questions():
            leaf, chain = self._resolve_chain(q, [])
            # leaf是链底的真正可问题
            if leaf and leaf not in [x[0] for x in leaves]:  # 避免重复
                sim = self.sim_tool.similarity(ctx_now, leaf, persist_q1=False, persist_q2=True)
                leaves.append((leaf, sim, chain))
        return leaves

    def _resolve_chain(self, question, chain, depth=0):
        if depth > self.max_depth:
            print(f"[警告] 依赖链过长：{'->'.join(chain+[question])}")
            return None, chain+[question]
        deps = self.state_manager.get_question_depends_on(question)
        if not deps:
            return question, chain+[question]
        pending_deps = [dep for dep in deps if not self.state_manager.is_question_completed(self.state_manager.get_question_by_id(dep))]
        if not pending_deps:
            return question, chain+[question]
        # 递归展开所有依赖
        for dep in pending_deps:
            leaf, sub_chain = self._resolve_chain(self.state_manager.get_question_by_id(dep), chain+[question], depth+1)
            if leaf:
                return leaf, sub_chain
        return None, chain+[question]

    def get_best_next_question(self, dialog_history):
        leaves = self.get_all_ready_leaves(dialog_history)
        if not leaves:
            return None
        # 选和历史最相关的那一题
        leaves_sorted = sorted(leaves, key=lambda x: x[1], reverse=True)
        return leaves_sorted[0][0]  # 返回最佳题目文本
    
if __name__ == "__main__":
    # 模拟MemorySurveyStateManager
    class DummyStateManager:
        def __init__(self):
            self.data = [
                {"id": "Q1", "question": "两周内是否持续情绪低落？", "depends_on": [], "status": None},
                {"id": "Q2", "question": "是否出现兴趣丧失？", "depends_on": ["Q1"], "status": None},
                {"id": "Q3", "question": "你是否常常觉得疲惫？", "depends_on": ["Q2"], "status": None},
                {"id": "Q4", "question": "最近体重有无明显变化？", "depends_on": [], "status": None}
            ]
            self.id2item = {d['id']: d for d in self.data}
        def get_pending_questions(self):
            return [d["question"] for d in self.data if d["status"] is None]
        def get_question_depends_on(self, question):
            for d in self.data:
                if d["question"] == question:
                    return d.get("depends_on", [])
            return []
        def is_question_completed(self, question):
            for d in self.data:
                if d["question"] == question:
                    return d.get("status") is not None
            return False
        def get_question_by_id(self, qid):
            return self.id2item.get(qid, {}).get("question")

    from utils.embedding_similarity import EmbeddingSimilarityTool
    sim_tool = EmbeddingSimilarityTool()

    state_mgr = DummyStateManager()
    selector = SurveySelector(state_mgr, sim_tool)

    # 测试用对话历史
    dialog_history = [
        {"role": "user", "content": "我最近很低落，经常觉得没劲。"},
        {"role": "assistant", "content": "请问你有兴趣丧失吗？"},
    ]

    print("\n== Case 1: 所有题都未完成 ==")
    best_q = selector.get_best_next_question(dialog_history)
    print("最佳下一题:", best_q)

    # 假如Q1完成（依赖解锁Q2链），Q4还是没依赖
    state_mgr.data[0]["status"] = True
    print("\n== Case 2: Q1已完成 ==")
    best_q2 = selector.get_best_next_question(dialog_history)
    print("最佳下一题:", best_q2)

    # 假如Q2也完成（依赖解锁Q3链）
    state_mgr.data[1]["status"] = True
    print("\n== Case 3: Q1,Q2已完成 ==")
    best_q3 = selector.get_best_next_question(dialog_history)
    print("最佳下一题:", best_q3)

    # 假如Q1~Q3全完成
    state_mgr.data[2]["status"] = True
    print("\n== Case 4: 只剩无依赖Q4 ==")
    best_q4 = selector.get_best_next_question(dialog_history)
    print("最佳下一题:", best_q4)

    # 全完成
    state_mgr.data[3]["status"] = True
    print("\n== Case 5: 全部完成 ==")
    best_q5 = selector.get_best_next_question(dialog_history)
    print("最佳下一题:", best_q5)
