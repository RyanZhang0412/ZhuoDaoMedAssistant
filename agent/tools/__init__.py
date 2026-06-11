"""agent.tools 包。

导入三个工具模块以触发 @register_tool 装饰器，把工具注册进 TOOL_REGISTRY。
（导入即注册 —— Agent.get_tool_schemas() 从全局注册表收集。）

注意：全部为本地工具，无任何联网插件（无 web_search/weather 等）。
"""

# 导入即注册（顺序无关）
from agent.tools import record_tools, recommend_tools, schedule_tools  # noqa: F401
from agent.tools.context import ToolContext, get_context, set_context
from agent.tools.schedule_tools import Scheduler

__all__ = ["ToolContext", "get_context", "set_context", "Scheduler"]
