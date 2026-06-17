# 卓道康复助手 ZhuoDaoMedAssistant

本地、原则上**不联网**的医疗康复辅助助手。参考 [百聆 bailing](https://github.com/wwbin2017/bailing)
的语音对话框架（ASR + VAD + LLM + TTS + Robot 协调层 + 本地 Agent），
但**去掉了所有联网插件**（web_search / weather / OpenManus），并新增医疗核心：
录入/读取患者病历，从既往病历用**规则引擎**推导康复训练模式。

## 设计理念

- **规则为主 + LLM 辅助**：康复训练方案由规则引擎（医生维护的 `config/rules.yaml`）
  确定性给出，可解释、可审计；LLM 只负责"读懂病历"和"把结论说人话"，**不参与拍板**。
- **病历与对话记忆严格隔离**：正式病历存 `data/patients/`，闲聊上下文存 `data/dialogues/`，
  LLM 闲聊内容绝不污染病历。
- **多 provider 可切换**：开发期用 API（显存不够时），移植服务器换本地部署只改 `config.yaml`。
- **不联网硬约束**：`offline: true` 时在进程最早期装上 socket 级禁网兜底。

## 架构

```
 语音输入 ──ASR──┐                                    ┌── TTS──> 语音输出
                ├─> Robot(协调层) ─> Agent(本地工具) ──┤
 文本输入 ──────┘        │              │             └── 文本回复
                         │              └─ record_tools  (病历 CRUD)
                  DialogueMemory        └─ recommend_tools(康复推荐)
                  (对话上下文)           └─ schedule_tools (训练提醒)
                                              │
                              medical/service (薄推荐门面)
                                病历 → RuleEngine(规则拍板) → LLM解释(可选)
```

## 目录结构

```
ZhuoDaoMedAssistant/
├── main.py                  # 入口：装配 + 文本 REPL（--text）
├── config/
│   ├── config.yaml          # 模块/LLM provider/路径
│   ├── rules.yaml           # ★康复训练规则库（医生维护，不碰代码）
│   └── system_prompt.txt    # Agent 系统提示
├── core/                    # 语音对话引擎
│   ├── robot.py             # 协调层
│   ├── asr/ vad/ tts/       # 可插拔语音模块（仅本地引擎）
│   └── llm/                 # ★LLMBase 统一抽象 + 多 provider 工厂
├── medical/                 # ★医疗核心
│   ├── repository.py        # 病历数据访问层（当前本地 JSON，后续可换数据库）
│   ├── rules.py             # 规则加载与匹配
│   └── service.py           # 薄推荐门面（规则 + 可选LLM解释）
├── agent/                   # 本地 Agent（无联网插件）
│   ├── agent.py             # function-calling 主循环 + 工具注册表
│   └── tools/               # record / recommend / schedule 工具
├── memory/dialogue_memory.py# 对话记忆（与病历隔离）
├── data/{patients,schedules,dialogues}/  # 本地数据
└── tests/                   # 规则引擎/仓库/agent 测试
```

## 安装

```bash
# 核心（必装）
pip install pyyaml pydantic jsonschema

# LLM provider 客户端（按所选 provider 装其一）
pip install openai          # provider=openai_compatible / ollama
# pip install anthropic     # provider=anthropic

# 可选：语音（不需要可跳过，--text 文本模式不依赖）
# pip install funasr onnxruntime sounddevice kokoro

# 可选：本地大模型（移植服务器时）
# pip install transformers torch   或   llama-cpp-python   或本地起 ollama

# 测试
pip install pytest
```

## 配置（provider 切换 / offline）

编辑 `config/config.yaml`：

- **开发期（你现在，显存不够用 API）**：
  ```yaml
  offline: false
  llm:
    provider: openai_compatible
    openai_compatible:
      base_url: https://api.deepseek.com/v1
      api_key: ${OPENAI_API_KEY}   # 从环境变量读，不写明文
      model: deepseek-chat
  ```
  设置环境变量：`export OPENAI_API_KEY=sk-xxx`（Windows: `set OPENAI_API_KEY=sk-xxx`）

- **移植服务器（本地部署）**：只改这两处，代码零改动
  ```yaml
  offline: true
  llm:
    provider: ollama          # 或 local
    ollama:
      base_url: http://127.0.0.1:11434/v1
      model: qwen2.5:7b
  ```

## 运行

```bash
# 文本对话（推荐，无需语音依赖与麦克风）
python main.py --text

# 聚焦某患者
python main.py --text --patient P001
```

## 维护康复规则（医生）

只改 `config/rules.yaml`，无需碰代码。每条规则：

```yaml
- name: 脑卒中-偏瘫早期-上肢被动训练
  priority: 100                    # 越大优先级越高
  conditions:                      # 多条件 AND
    diagnosis: 脑卒中
    affected_limb: 上肢
    muscle_strength: "<=2"        # 支持 < <= > >= == 和范围 "60-80"、多选 [上肢,下肢]
  recommend:
    mode: 被动关节活动训练
    frequency: 每日2次
    intensity: 低
    precautions: [需康复治疗师全程陪同]
    goal: 维持关节活动度，预防挛缩
```

条件字段必须是病历已知字段（见 `medical/repository.py` 的 `RULE_CONDITION_FIELDS`），
否则加载报错，防止规则与病历字段漂移。

## 数据与隐私

- 病历存本地 JSON：`data/patients/{patient_id}.json`，开发期保持最小实现，后续可替换为数据库。
- 全程不联网（`offline: true` 时硬性拦截一切非本地网络出口）。
- API key 从环境变量读，不硬编码进配置。

## 测试

```bash
python -m pytest tests/ -q
```

覆盖：规则引擎确定性匹配、"结论来自规则而非 LLM"边界、病历 CRUD/乐观锁、offline 禁网守卫。
