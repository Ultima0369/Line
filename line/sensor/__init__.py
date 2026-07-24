"""
传感器模块 — 连接物理世界与认知框架。

架构:
    base.Sensor       ← 抽象基类，所有传感器继承
    manager.Manager   ← 传感器管理器，统一注册/轮询/聚合
    mock.MockSensor   ← 模拟传感器（无硬件时开发用）
    temperature, pressure, light, audio, system  ← 具体驱动

用法:
    from line.sensor import SensorManager
    manager = SensorManager()
    await manager.scan()
    data = await manager.read_all()
"""
