"""
Line 核心路径测试。

无第三方测试框架依赖，纯 stdlib + assert，可直接:
    python tests/run_tests.py
也可被 pytest 收集（test_ 前缀）:
    python -m pytest tests/ -q
"""

import asyncio
import os
import sys
from pathlib import Path

# 让测试能 import 顶层 line 包
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_singleton_no_state_leak():
    """Config 单例：新实例不应继承上一次的 _data。"""
    from line.config import Config
    # 清掉单例，模拟全新进程
    Config._instance = None
    c1 = Config()
    c1._data = {"stale": True}
    # 再取实例（单例已存在），但 _data 是实例属性，不应跨实例残留
    c2 = Config()
    assert c2 is c1, "单例应返回同一对象"
    assert c2._data == {"stale": True}, "同一实例，数据应保留"
    # 真正的泄漏测试：重置单例后新实例必须干净
    Config._instance = None
    c3 = Config()
    assert c3._data == {}, f"新单例实例不应继承旧数据: {c3._data}"
    Config._instance = None


def test_encode_sensor_numeric():
    """数值传感器编码为紧凑串。"""
    from line.edge.compressor import SemanticProtocol as SP
    data = {"temperature": [{"value": 26.5, "unit": "°C"}],
            "humidity": [{"value": 58.2, "unit": "%"}]}
    out = SP.encode_sensor_data(data)
    assert out == "tmp:26.5|hum:58.2", out


def test_encode_sensor_composite_system():
    """system 传感器值是 dict，必须按子指标展开，不能塞 dict repr 进串。"""
    from line.edge.compressor import SemanticProtocol as SP
    data = {"system": [{"value": {"cpu_temp": 27.9, "cpu_percent": 2.1,
                                  "memory_percent": 46.9}, "unit": "composite"}]}
    out = SP.encode_sensor_data(data)
    assert "cputmp:27.9" in out, out
    assert "cpupct:2.1" in out, out
    assert "mempct:46.9" in out, out
    assert "{" not in out, f"composite 值不应泄漏 dict repr: {out}"


def test_upstream_build_structure():
    """上行数据包结构完整，传感器字段是紧凑串。"""
    from line.edge.compressor import SemanticProtocol as SP
    up = SP.build_upstream(
        user_input="热死了",
        sensor_data={"temperature": [{"value": 38.5, "unit": "°C"}]},
        attention_state={"attention_level": 0.7, "status": "active"},
        context={"summary": "用户问温度", "turn_count": 3},
    )
    assert up["user"] == "热死了"
    assert up["sensor"] == "tmp:38.5", up["sensor"]
    assert up["attention"] == "l:0.7|s:active", up["attention"]
    assert up["version"] == SP.VERSION


def test_user_input_truncation():
    """超长 user 输入应截断到 2000 字符。"""
    from line.edge.compressor import SemanticProtocol as SP
    up = SP.build_upstream(user_input="x" * 3000)
    assert len(up["user"]) == 2000


def test_attention_high_temp_hard_fire():
    """温度越过物理阈值 → priority=0.9 → 硬触发 alert。"""
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor({"interrupt_threshold": 0.7})
    # 初始化不依赖 async 副作用（local_model 占位）
    edge.evaluate_sensor_batch({"temperature": [{"value": 39.0, "unit": "°C"}]})
    alerts = edge.drain_alerts()
    assert alerts, "高温 39°C 应硬触发 alert"
    assert "temperature" in alerts[0]


def test_attention_normal_temp_no_fire():
    """正常温度不触发。"""
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor({"interrupt_threshold": 0.7})
    edge.evaluate_sensor_batch({"temperature": [{"value": 24.0, "unit": "°C"}]})
    assert edge.drain_alerts() == [], "正常温度不应触发"


def test_attention_system_metric_evaluated():
    """system 传感器的子指标必须被注意力评估，不能被 skip。"""
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor({"interrupt_threshold": 0.7})
    # memory 90% 超过 humidity 阈值映射？不会——阈值表只认 temperature/humidity/pressure。
    # 但 cpu_temp 若极高应触发。这里验证 system 子指标确实进入了评估流（有信号记录）。
    edge.evaluate_sensor_batch({"system": [{"value": {"cpu_temp": 95.0,
                                                       "cpu_percent": 99.0,
                                                       "memory_percent": 90.0},
                                             "unit": "composite"}]})
    ctx = edge.attention.get_context_summary()
    assert ctx["signal_count"] > 0, f"system 子指标未被评估: {ctx}"
    # cpu_temp 95 高温不在阈值表里，但应至少被记入信号历史
    assert "system.cpu_temp" in ctx.get("dominant_sources", {}), ctx


def test_drain_clears_alerts():
    """drain 后队列应清空。"""
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor({})
    edge.evaluate_sensor_batch({"temperature": [{"value": 39.0, "unit": "°C"}]})
    assert edge.drain_alerts()
    assert edge.drain_alerts() == []


def test_attention_drives_temperature():
    """认知架构落地点：注意力状态驱动云端采样温度。

    active（高注意力）→ 降温度聚焦；calm（低注意力）→ 升温度发散。
    不再只塞一句话进 prompt，而是改变推理参数。
    """
    from line.cloud.bridge import CloudBridge
    b = CloudBridge({"api_key": "fake"})

    # 高注意力 level=0.9 → 应低温度（聚焦）
    up_active = {"user": "x", "attention": "l:0.9|s:active"}
    t_active = b._sample_params_from_attention(up_active)["temperature"]
    # 低注意力 level=0.1 → 应高温度（发散）
    up_calm = {"user": "x", "attention": "l:0.1|s:calm"}
    t_calm = b._sample_params_from_attention(up_calm)["temperature"]

    assert t_active < t_calm, f"高注意力应更低温度: active={t_active} calm={t_calm}"
    assert 0.3 <= t_active <= 1.0, t_active
    assert 0.3 <= t_calm <= 1.0, t_calm
    # level=1.0 → temp 0.3 ; level=0.0 → temp 1.0
    assert b._sample_params_from_attention({"user": "x", "attention": "l:1.0|s:active"})["temperature"] == 0.3
    assert b._sample_params_from_attention({"user": "x", "attention": "l:0.0|s:calm"})["temperature"] == 1.0


def test_attention_level_parse_fallback():
    """注意力串解析失败时回中性 0.5，不崩。"""
    from line.cloud.bridge import CloudBridge
    b = CloudBridge({})
    assert CloudBridge._parse_attention_level("l:0.7|s:active") == 0.7
    assert CloudBridge._parse_attention_level("") == 0.5
    assert CloudBridge._parse_attention_level("garbage") == 0.5
    assert CloudBridge._parse_attention_level("l:notanumber") == 0.5


def test_proactive_turn_triggers_on_threshold():
    """环境阈值越界 + 冷却期外 → 主动触发，返回 upstream 包。"""
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor({"proactive_cooldown": 60.0})
    up = edge.maybe_proactive_turn({"temperature": [{"value": 39.0, "unit": "°C"}]}, now=1000.0)
    assert up is not None, "高温+冷却外应主动触发"
    assert up.get("meta", {}).get("proactive") is True, "应标记 proactive"
    assert "[主动环境推理]" in up["user"], up["user"]


def test_proactive_turn_cooldown_blocks():
    """冷却期内重复异常不再触发，省 token。"""
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor({"proactive_cooldown": 60.0})
    # t=1000 触发一次
    assert edge.maybe_proactive_turn({"temperature": [{"value": 39.0, "unit": "°C"}]}, now=1000.0) is not None
    # t=1030 在 60s 冷却内 → 不触发
    assert edge.maybe_proactive_turn({"temperature": [{"value": 39.0, "unit": "°C"}]}, now=1030.0) is None
    # t=1061 过冷却 → 再次触发
    assert edge.maybe_proactive_turn({"temperature": [{"value": 39.0, "unit": "°C"}]}, now=1061.0) is not None


def test_proactive_turn_no_trigger_on_normal():
    """正常环境不主动触发。"""
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor({})
    assert edge.maybe_proactive_turn({"temperature": [{"value": 24.0, "unit": "°C"}]}, now=1000.0) is None


def test_proactive_turn_no_trigger_empty():
    """空数据不触发。"""
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor({})
    assert edge.maybe_proactive_turn({}, now=1000.0) is None


def test_cloudbridge_ask_no_key_degrades():
    """无 API key 时 ask 应优雅降级，不抛异常。"""
    import asyncio
    from line.cloud.bridge import CloudBridge
    from line.edge.compressor import SemanticProtocol as SP

    async def run():
        b = CloudBridge({})  # 无 key
        await b.initialize()
        up = SP.build_upstream(user_input="hi")
        r = await b.ask(upstream=up)
        assert "API 密钥未配置" in r["content"], r["content"]
        await b.shutdown()

    asyncio.run(run())


def test_cloudbridge_retries_on_5xx(monkeypatch_like=None):
    """429/5xx 应重试，4xx 不应重试（验证 max_retries 被读）。"""
    import asyncio
    import httpx
    from line.cloud.bridge import CloudBridge
    from line.edge.compressor import SemanticProtocol as SP

    calls = {"n": 0}

    async def run():
        b = CloudBridge({"max_retries": 2, "api_key": "fake"})
        await b.initialize()
        # 替换 client.post 为总是返回 503 的桩
        class FakeResp:
            status_code = 503
            text = "service unavailable"
            def raise_for_status(self):
                raise httpx.HTTPStatusError("503", request=None, response=self)
            def json(self): return {}

        async def fake_post(url, json=None):
            calls["n"] += 1
            return FakeResp()

        b._client.post = fake_post  # type: ignore
        up = SP.build_upstream(user_input="hi")
        r = await b.ask(upstream=up)
        # max_retries=2 → 1 次首试 + 2 次重试 = 3 次
        assert calls["n"] == 3, f"应重试到 3 次, 实际 {calls['n']}"
        assert "失败" in r["content"]
        await b.shutdown()

    asyncio.run(run())


def _run_all():
    """收集并运行所有 test_ 函数。"""
    g = dict(globals())
    tests = [(name, fn) for name, fn in g.items() if name.startswith("test_") and callable(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  通过 {passed} | 失败 {failed} | 共 {passed+failed}")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
