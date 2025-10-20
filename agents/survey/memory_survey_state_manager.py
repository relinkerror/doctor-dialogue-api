from typing import List, Dict, Any, Optional

class MemorySurveyStateManager:
    """
    负责维护所有问卷及题目的作答状态，支持id高效查找、依赖、remark和进度统计。

    survey_states: {
        survey_id: {
            'items': [
                {
                    'id': "MDD_Q1",
                    'question': "……",
                    'depends_on': ["MDD_Q0"],   # List[str]
                    'remarks': ["……"],         # List[str]
                    'status': True or False or None
                },
                ...
            ],
            'completed': False
        },
        ...
    }
    id2item: { "MDD_Q1": item_dict, ... }  # 全局唯一id索引
    """
    def __init__(self):
        self.survey_states: Dict[str, Dict[str, Any]] = {}
        self.id2item: Dict[str, Dict[str, Any]] = {}

    def initialize_states(self, surveys: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        初始化问卷状态。
        Args:
            surveys: Dict[str, List[Dict]]
                结构示例:
                {
                  "MDD": [
                    {'id':"MDD_Q1", 'question':"……", 'depends_on':[], 'remarks':[], 'status':None},
                    …
                  ],
                  ...
                }
        """
        self.survey_states.clear()
        self.id2item.clear()
        for survey_id, items in surveys.items():
            copied_items = [dict(item) for item in items]
            for item in copied_items:
                self.id2item[item['id']] = item
            self.survey_states[survey_id] = {
                'items': copied_items,
                'completed': False
            }

    def get_pending_questions(self) -> List[str]:
        """
        获取所有未完成（status=None）题目的文本列表。
        Returns:
            List[str]: ["题目文本1", "题目文本2", ...]
        """
        return [
            item['question']
            for state in self.survey_states.values()
            for item in state['items']
            if item.get('status') is None
        ]

    def get_pending_surveys(self) -> List[str]:
        """
        获取所有未完成的问卷ID列表。
        Returns:
            List[str]: ["MDD", "PDD", ...]
        """
        return [sid for sid, s in self.survey_states.items() if not s['completed']]

    def get_question_context(self, question: str) -> str:
        """
        获取题目的全部上下文（依赖+本题状态+备注），用于Prompt构建。
        Args:
            question: str, 问题文本
        Returns:
            str, 详尽格式说明如下:
            Context:
            - MDD_Q1: …… [status: True] Remarks: ……
            Question MDD_Q2: …… [status: False]
            Remarks: ……
        """
        for state in self.survey_states.values():
            for item in state['items']:
                if item['question'] != question:
                    continue
                lines: List[str] = []
                deps = item.get('depends_on', [])
                if deps:
                    lines.append("Context:")
                    for dep_id in deps:
                        dep = self.id2item.get(dep_id)
                        if not dep:
                            continue
                        dep_remarks = dep.get('remarks', [])
                        dep_remark_str = f" Remarks: {', '.join(dep_remarks)}" if dep_remarks else ""
                        lines.append(
                            f"- {dep['id']}: {dep['question']} [status: {dep.get('status')}]"
                            + dep_remark_str
                        )
                cur_status = item.get('status')
                lines.append(f"Question {item['id']}: {item['question']} [status: {cur_status}]")
                own_remarks = item.get('remarks', [])
                if own_remarks:
                    lines.append(f"Remarks: {', '.join(own_remarks)}")
                return "\n".join(lines)
        return "(未找到该问题)"

    def get_question_depends_on(self, question: str) -> List[str]:
        """
        获取某题所有依赖id列表。
        Args:
            question: str
        Returns:
            List[str] 如 ["MDD_Q1", "MDD_Q2"]
        """
        for state in self.survey_states.values():
            for item in state['items']:
                if item['question'] == question:
                    return item.get('depends_on', [])
        return []

    def get_question_by_id(self, qid: str) -> Optional[str]:
        """
        根据题目唯一id返回题目原文。
        Args:
            qid: str (如'MDD_Q3')
        Returns:
            str (题目原文) 或 None
        """
        item = self.id2item.get(qid)
        return item['question'] if item else None

    def update_states(self, survey_name: str, results: List[Dict[str, Any]]) -> None:
        """
        批量更新题目的status/remarks。
        Args:
            survey_name: str
            results: List[Dict]，每项结构如下:
                {
                    'id': "MDD_Q2",
                    'status': True/False/None,
                    'remarks': ["xxx", ...]  # 可选，也可为"remark": "xxx"
                }
        """
        MAX_REMARK_LENGTH = 100
        MAX_REMARK_COUNT  = 2
        state = self.survey_states.get(survey_name)
        if not state:
            return
        items = state['items']
        id_map = {it['id']: it for it in items}
        text_map = {it['question'].strip(): it['id'] for it in items}
        for res in results:
            rid = res.get('id') or text_map.get(res.get('question', '').strip())
            if not rid or rid not in id_map:
                continue
            item = id_map[rid]
            if item.get('status') is not None:
                continue
            raw = res.get('remarks') or res.get('remark')
            if raw:
                new_list = raw if isinstance(raw, list) else [raw]
                for remark in new_list:
                    truncated = remark[:MAX_REMARK_LENGTH]
                    if truncated not in item.setdefault('remarks', []):
                        item['remarks'].append(truncated)
                if len(item['remarks']) > MAX_REMARK_COUNT:
                    item['remarks'] = item['remarks'][-MAX_REMARK_COUNT:]
            new_status = res.get('status')
            if new_status in (True, False):
                item['status'] = new_status
        state['completed'] = all(it.get('status') is not None for it in items)

    def is_diagnosis_completed(self) -> bool:
        """
        判断所有问卷是否已全部作答。
        Returns:
            bool
        """
        return all(s['completed'] for s in self.survey_states.values())

    def get_diagnosis_result(self) -> Any:
        """
        返回完成率最高的问卷id。
        Returns:
            str 或 None
        """
        candidates = []
        for name, s in self.survey_states.items():
            if not s['completed']:
                continue
            items = s['items']
            total = len(items)
            trues = sum(1 for it in items if it['status'] is True)
            if total:
                candidates.append((name, trues/total, trues))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        return candidates[0][0]

    def is_question_completed(self, question: str) -> bool:
        """
        判断某题是否已作答。
        Args:
            question: str
        Returns:
            bool
        """
        if not question or not question.strip():
            return False
        q_norm = question.strip()
        for state in self.survey_states.values():
            for item in state['items']:
                if item.get('question', '').strip() == q_norm:
                    return item.get('status') is not None
        return False

    def get_all_answers(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取所有问卷的结构化答题结果。
        Returns:
            Dict[str, List[Dict]]
            结构示例:
            {
                "MDD": [
                    {
                        "id": "MDD_Q1",
                        "question": "...",
                        "status": True,
                        "remarks": ["remark1", ...]
                    },
                    ...
                ],
                ...
            }
        """
        all_states = {}
        for survey_name, state in self.survey_states.items():
            items = state.get('items', [])
            all_states[survey_name] = [
                {
                    'id': item['id'],
                    'question': item['question'],
                    'status': item.get('status'),
                    'remarks': item.get('remarks', [])
                }
                for item in items
            ]
        return all_states

    def get_progress(self) -> str:
        """
        获取问卷整体进度。
        Returns:
            str, 例: "已完成2/6问卷，共8/60题完成"
        """
        n_total = n_done = 0
        n_survey = len(self.survey_states)
        n_survey_done = 0
        for name, s in self.survey_states.items():
            items = s['items']
            n_total += len(items)
            n_done += sum(1 for it in items if it.get('status') is not None)
            if all(it.get('status') is not None for it in items):
                n_survey_done += 1
        return f"已完成{n_survey_done}/{n_survey}问卷，共{n_done}/{n_total}题完成"

# =============== main 测试用例 ===============

if __name__ == "__main__":
    surveys = {
        "MDD": [
            {"id": "MDD_Q1", "question": "两周内是否持续情绪低落？", "depends_on": [], "remarks": [], "status": None},
            {"id": "MDD_Q2", "question": "是否出现兴趣丧失？", "depends_on": ["MDD_Q1"], "remarks": [], "status": None}
        ],
        "PDD": [
            {"id": "PDD_Q1", "question": "情绪低落是否持续超过2年？", "depends_on": [], "remarks": [], "status": None}
        ]
    }
    mgr = MemorySurveyStateManager()
    mgr.initialize_states(surveys)

    print("【全部未完成问题】:", mgr.get_pending_questions())
    print("【MDD_Q1上下文】:")  # context输出
    print(mgr.get_question_context("两周内是否持续情绪低落？"))

    # id查题
    print("【用id查题】MDD_Q2:", mgr.get_question_by_id("MDD_Q2"))
    print("【用不存在id查题】XXX_Q99:", mgr.get_question_by_id("XXX_Q99"))

    # 查依赖
    print("【MDD_Q2的依赖id列表】:", mgr.get_question_depends_on("是否出现兴趣丧失？"))

    # remark追加
    mgr.update_states("PDD", [
        {"id": "PDD_Q1", "status": True, "remarks": ["症状持续2年以上", "用户主观报告"]}
    ])
    print("【PDD_Q1上下文】:")
    print(mgr.get_question_context("情绪低落是否持续超过2年？"))

    # 不存在问题
    print("【不存在问题上下文】:")
    print(mgr.get_question_context("不存在的问题"))

    # 批量作答+进度
    mgr.update_states("MDD", [
        {"id": "MDD_Q1", "status": True, "remarks": "用户报告情绪低落"},
        {"id": "MDD_Q2", "status": False, "remarks": "兴趣尚可"}
    ])
    print("【进度】:", mgr.get_progress())
    print("【已完成？】:", mgr.is_diagnosis_completed())
    print("【全部答案】:")
    import json
    print(json.dumps(mgr.get_all_answers(), ensure_ascii=False, indent=2))
