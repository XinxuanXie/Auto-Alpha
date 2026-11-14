# 文档索引

本目录按研究阶段组织设计文档。共享基础设施只维护一份，各阶段文档保留当时的设计口径，避免把历史实验与当前方案混写。

## 共享基础设施

- [`shared/data_backtest_pipeline.md`](shared/data_backtest_pipeline.md)：数据接口、因子清洗、中性化与统一回测。
- [`shared/factor_dsl.md`](shared/factor_dsl.md)：FactorSpec、受限公式 AST、算子与校验规则。
- [`shared/factor_knowledge_base.md`](shared/factor_knowledge_base.md)：本地因子知识库内容与用途。

## Agent Workflow V1

第一版开放式研究工作流：四个 Specialist Research Agent、两阶段研究、FactorSpec、LangGraph Manager、回测与 SQLite Memory。

- [`agent_workflow_v1/manager_workflow_notes.md`](agent_workflow_v1/manager_workflow_notes.md)
- [`agent_workflow_v1/outer_manager_agent.md`](agent_workflow_v1/outer_manager_agent.md)
- [`agent_workflow_v1/two_stage_research.md`](agent_workflow_v1/two_stage_research.md)

## EvoAlpha Workflow

独立的种子池、Refinement、Crossover、Mutation、相关性过滤和多代 Population Update 实验。该阶段保留为对照方案，不代表当前主研究方向。

- [`evoalpha_workflow/evoalpha_single_factor_design.md`](evoalpha_workflow/evoalpha_single_factor_design.md)
- [`evoalpha_workflow/implementation_backlog.md`](evoalpha_workflow/implementation_backlog.md)
- [`evoalpha_workflow/research_improvement_ideas.md`](evoalpha_workflow/research_improvement_ideas.md)

## Hypothesis Workflow V2

当前 V2.1 方向：把 LLM 能力集中在金融洞察的批判性深化，固定四个独立研究方向，在假设与公式之间加入 Measurement Plan，并使用确定性代码完成谱系检查、回测和记录。

- [`hypothesis_workflow_v2/design.md`](hypothesis_workflow_v2/design.md)
- [`hypothesis_workflow_v2/contracts_and_state_graph.md`](hypothesis_workflow_v2/contracts_and_state_graph.md)：Pydantic 数据契约、Artifact 设计与固定 Manager 状态图。
- [`hypothesis_workflow_v2/file_guide.md`](hypothesis_workflow_v2/file_guide.md)：V2 每个源文件的职责、调用关系与 checkpoint 恢复方式。
- [`hypothesis_workflow_v2/implementation_log.md`](hypothesis_workflow_v2/implementation_log.md)：按日期记录实现、真实运行与后续事项。
