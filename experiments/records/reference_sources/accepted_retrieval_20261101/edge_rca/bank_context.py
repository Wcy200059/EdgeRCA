"""Public Bank vocabulary, independent of individual questions and labels.

Candidate names come from OpenRCA's basic_prompt_Bank.py:cand. Additional
observation identities are routing metadata; no docker-to-service alias is
asserted. This module does not read telemetry, query rows or scoring points.
"""

BANK_COMPONENTS = (
    "apache01", "apache02", "Tomcat01", "Tomcat02", "Tomcat04", "Tomcat03",
    "MG01", "MG02", "IG01", "IG02", "Mysql01", "Mysql02", "Redis01", "Redis02",
)

BANK_REASONS = (
    "high CPU usage", "high memory usage", "network latency", "network packet loss",
    "high disk I/O read usage", "high disk space usage", "high JVM CPU load",
    "JVM Out of Memory (OOM) Heap",
)


def public_bank_context() -> dict:
    """Return a fresh copy of shared background, never a case-specific answer."""
    return {
        "benchmark": "Bank",
        "timezone": "Asia/Shanghai",
        "root_cause_components": list(BANK_COMPONENTS),
        "root_cause_reasons": list(BANK_REASONS),
        "additional_observation_nodes": ["dockerA1", "dockerA2", "dockerB1", "dockerB2"],
        "interpretation": [
            "候选是所有Bank题通用的公开范围，不是当前题答案；证据不足可保留未知。",
            "docker节点仍可能提供调用关系；没有已核实映射，不能擅自改名为Mysql或Redis。",
            "ServiceTest业务流独立汇总，没有可靠映射时不能归到某个设备。",
            "Bank原始调用表没有成功/失败状态；慢调用、未匹配父调用均不等于调用失败。",
        ],
        "candidate_source": "OpenRCA/rca/baseline/rca_agent/prompt/basic_prompt_Bank.py:cand",
    }
