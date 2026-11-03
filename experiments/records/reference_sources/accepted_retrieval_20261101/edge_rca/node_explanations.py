"""Short, question-aware interpretations kept separate from tool facts."""

from copy import deepcopy


FIELDS = {"meaning": 40, "can_answer": 80, "cannot_answer": 80, "next_check": 60}

INSTRUCTION = """\n本次使用详细填写表。每个items还必须填写interpretation对象，严格含4个短句：
meaning：本条观测在表达什么（最多40字）。
can_answer：依据本条材料，能回答原问题的哪一部分；若材料反对某个猜测就明确写出（最多80字）。
cannot_answer：原问题还有哪部分不能回答，为什么；说明是缺少数据、只有摘要、工具未计算还是尚不能建立关联（最多80字）。
next_check：仍需什么证据或对照；已经足够可写无需补充（最多60字）。
原问题在investigation_request中，问题可能包含多个判断，不要只输出一句证据不足。
例如CPU采样不高，可以说明采样不支持持续高CPU；不能因此确认其他原因或排除采样间隙短时变化。
工具无法满足原问题时，仍须写已知的部分与未解决的部分，不能把未解决的问题丢掉。
所有数字、时间、次数由程序附带，不重算、不编造；只解释本页提供的材料。不因问题暗示OOM就认定OOM。
没提供相关指标/日志/调用就明确缺少，不能假定存在；没有日志不表示没有故障。
没有investigation_request时，can_answer写能确认的本地现象，不能自行确定全局根因。
解释是待核实的模型意见，不能冒充原始事实。格式片段：
"interpretation":{"meaning":"描述观测含义","can_answer":"能够回答的部分","cannot_answer":"还不能回答及原因","next_check":"需要的下一份材料或无需补充"}。
"""


def validate(value):
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError("interpretation needs meaning/can_answer/cannot_answer/next_check")
    for key, maximum in FIELDS.items():
        if not isinstance(value[key], str) or not 1 <= len(value[key].strip()) <= maximum:
            raise ValueError(f"interpretation.{key} needs 1..{maximum} characters")
    return deepcopy(value)


def describe(value):
    return "；".join(label + value[key] for key, label in (
        ("meaning", "现象："), ("can_answer", "能回答："),
        ("cannot_answer", "还不能回答："), ("next_check", "下一步：")))
