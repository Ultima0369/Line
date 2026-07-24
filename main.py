"""
Line — 双脑异构体认知框架 · 主入口

启动命令:
    python main.py              # 交互式双脑对话
    python main.py --config     # 查看配置
    python main.py --sensors    # 仅测试传感器

环境变量:
    DEEPSEEK_API_KEY            # DeepSeek API 密钥（推荐方式，不写进文件）
"""

import asyncio
import sys
import os
import time
import logging
from pathlib import Path

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent))

from line.config import Config
from line.utils.logger import setup_logger
from line.utils.helpers import get_system_info, check_hardware_capability

logger = logging.getLogger(__name__)


async def interactive_mode():
    """交互式双脑对话模式。"""

    print("""
    ╔══════════════════════════════════════╗
    ║     Line — 双脑异构体认知框架        ║
    ║     本地感官皮层 + 云端新皮层        ║
    ╚══════════════════════════════════════╝
    """)

    # 加载配置
    config = Config()
    try:
        config.load()
    except FileNotFoundError as e:
        logger.warning(str(e))
        logger.warning("将以本地模式运行（无云端推理，仅传感器和本地处理）")

    issues = config.validate()
    if issues:
        for issue in issues:
            logger.warning(f"⚠️ {issue}")
        if any("api_key" in i for i in issues):
            api_key = input("\n或者直接输入 API Key（输入后回车）: ").strip()
            if api_key:
                os.environ["DEEPSEEK_API_KEY"] = api_key
                try:
                    config.load()  # 重新加载（环境变量会被读取）
                except FileNotFoundError:
                    pass
            else:
                print("  → 未输入 API Key，将以本地模式运行（无云端推理）")

    # 初始化传感器
    from line.sensor.manager import SensorManager
    sensor_manager = SensorManager()

    sensor_config = config.get("sensor", default={})
    if sensor_config.get("enabled", False) or sensor_config.get("mock", True):
        sensor_count = await sensor_manager.scan(sensor_config)
        logger.info(f"📡 传感器就绪: {sensor_count} 个")

    # 初始化边缘处理器
    from line.edge.processor import EdgeProcessor
    edge = EdgeProcessor(config.get("edge", default={}))
    await edge.initialize()

    # 初始化云端桥接器
    from line.cloud.bridge import CloudBridge
    cloud = CloudBridge(config.get("cloud", default={}))
    await cloud.initialize(config.get("cloud", default={}))

    # 连接边缘与云端
    async def upstream_handler(upstream: dict):
        """当边缘处理器产生上行数据包时，自动转发到云端。

        upstream 是 SemanticProtocol.build_upstream 的返回值，
        bridge 据此拼带环境上下文的 system prompt。
        """
        if not upstream.get("user", "").strip():
            return

        # 调用云端（消费整个 upstream）
        cloud_response = await cloud.ask(upstream=upstream)

        # 接收云端回复
        response = await edge.receive_downstream(cloud_response)

        # 显示回复
        print()

        # 如果有推理过程，折叠显示
        reasoning = cloud_response.get("reasoning", "")
        if reasoning:
            print(f"  ┌─ 🤔 推理过程 {'─' * 40}")
            for line in reasoning.strip().split("\n"):
                print(f"  │ {line}")
            print(f"  └{'─' * 50}")

        print(f"\n  🤖 {response}")

        # 显示延迟和用量（含注意力驱动的采样温度）
        latency = cloud_response.get("latency", 0)
        usage = cloud_response.get("usage", {})
        temp = cloud_response.get("temperature")
        temp_str = f" | T:{temp}" if temp is not None else ""
        if usage:
            print(f"\n  ── ({latency}s | 输入: {usage.get('prompt_tokens', '?')}t | 输出: {usage.get('completion_tokens', '?')}t{temp_str})")

    edge.on_upstream(upstream_handler)

    # 主动推理队列：环境异常触发的云端回复排这里，对话循环 drain 显示。
    # 不在轮询协程里直接打印——会和 input() 抢终端。排进队列，等用户回车间隙显示。
    proactive_queue: asyncio.Queue = asyncio.Queue()

    # 传感器轮询 → 注意力评估 → 决定是否主动触发云端推理
    async def attention_on_batch(batch):
        edge.evaluate_sensor_batch(batch)
        # maybe_proactive_turn 内部有冷却约束，不会每轮都触发
        proactive_upstream = edge.maybe_proactive_turn(batch, time.time())
        if proactive_upstream is not None:
            asyncio.create_task(_run_proactive(proactive_upstream))

    async def _run_proactive(upstream: dict):
        """独立协程：跑主动云端推理，结果入队。不阻塞对话循环。"""
        try:
            cloud_response = await cloud.ask(upstream=upstream)
            await edge.receive_downstream(cloud_response)
            await proactive_queue.put(cloud_response)
        except Exception as e:
            logger.error(f"主动推理失败: {e}")

    sensor_manager.on_batch(attention_on_batch)

    # 启动传感器轮询
    if sensor_config.get("enabled", False) or sensor_config.get("mock", True):
        sensor_manager.start_polling(interval=5.0)
        logger.info("🔄 传感器轮询已启动 (每5秒, 注意力评估 + 主动推理已接入)")

    print("\n  🤖 双脑就绪。输入你的消息开始对话。")
    print("  📝 输入 /help 查看命令  |  /exit 退出\n")

    # 对话循环
    try:
        while True:
            # 注入待处理的环境事件（注意力调度产出）
            alerts = edge.drain_alerts()
            if alerts:
                print()
                for a in alerts:
                    print(f"  {a}")

            # 显示主动推理结果（环境异常触发的云端回复，非阻塞）
            while not proactive_queue.empty():
                try:
                    resp = proactive_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                print()
                print("  ┌─ 🧠 主动环境推理（小脑主动叫醒大脑）")
                print(f"  │ {resp.get('content', '')[:300]}")
                print(f"  └─ (T:{resp.get('temperature', '?')})")
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("  👤 ")
                )
            except (EOFError, KeyboardInterrupt):
                print()
                break

            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                break

            if user_input.lower() == "/help":
                print("""
  📋 命令列表:
    /help       显示此帮助
    /sensors    显示传感器数据
    /status     显示系统状态
    /clear      清空对话上下文
    /exit       退出
                """)
                continue

            if user_input.lower() == "/sensors":
                data = await sensor_manager.read_all()
                if data:
                    from line.utils.helpers import format_sensor_for_display
                    print(f"\n  📡 {format_sensor_for_display(data)}")
                else:
                    print("\n  📡 无传感器数据")
                continue

            if user_input.lower() == "/status":
                info = get_system_info()
                hw = check_hardware_capability()
                edge_status = edge.get_status()
                print(f"""
  📊 系统状态:
    OS: {info['os']} {info['os_version']}
    Python: {info['python_version']}
    硬件: {hw['reason']}
    对话轮次: {edge_status['turn_count']}
    注意力水平: {edge_status.get('attention', {}).get('attention_level', 0)}
    传感器: {sensor_manager.sensor_count} 个
                """)
                continue

            if user_input.lower() == "/clear":
                edge.context.clear()
                print("\n  🧹 上下文已清空")
                continue

            # 正常处理
            sensor_data = await sensor_manager.read_all() if sensor_manager.sensor_count > 0 else None
            await edge.process(user_input, sensor_data)

    finally:
        sensor_manager.stop_polling()
        await sensor_manager.shutdown_all()
        await cloud.shutdown()
        print("\n  👋 再见\n")


def config_mode():
    """查看和验证配置。"""
    config = Config()
    try:
        config.load()
        import yaml
        print(yaml.dump(config.data, default_flow_style=False, allow_unicode=True))
        print("---")
        issues = config.validate()
        if issues:
            print("配置问题:")
            for i in issues:
                print(f"  ⚠️ {i}")
        else:
            print("✅ 配置验证通过")
    except FileNotFoundError:
        print("❌ 未找到配置文件。")
        print("   请将 config.example.yaml 复制为 config.yaml 并填写 API Key。")


async def sensor_test_mode():
    """仅测试传感器模式。"""
    from line.sensor.manager import SensorManager
    from line.utils.helpers import format_sensor_for_display

    manager = SensorManager()
    count = await manager.scan({"mock": True, "system": {"enabled": True}})
    print(f"📡 已注册 {count} 个传感器")

    print("\n读取传感器数据...")
    data = await manager.read_all()
    for sensor_type, readings in data.items():
        for r in readings:
            d = r.to_dict()
            print(f"  {d['sensor_type']}: {d['value']} {d['unit']} (置信度: {d['confidence']})")

    print("\n持续读取 5 次（每次间隔 2 秒）...")
    for i in range(5):
        await asyncio.sleep(2)
        data = await manager.read_all()
        print(f"  [{i+1}] {format_sensor_for_display(data)}")

    await manager.shutdown_all()


def main():
    # 初始化日志（必须在任何模块使用 logger 之前）
    setup_logger(level="INFO", log_file="line.log", console=True)

    # 解析参数
    args = sys.argv[1:]

    if "--config" in args:
        config_mode()
    elif "--sensors" in args:
        asyncio.run(sensor_test_mode())
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
