# 历年 BraTS METS validation 排行榜及提交可用性核查

复核日期：2026-11-17，北京时间。范围是本项目的脑转移瘤分割（METS），重点核查2023、2024、2025；不把成人胶质瘤、儿童肿瘤或其他赛道的开放入口算成METS入口。本次仅公开页面和官方只读API，无上传预测、提交比赛、服务器/VPN访问或依赖安装。

## 1. 结论

**三年METS旧validation成绩仍可浏览，但当前原文件提交队列均没有开放轮次，不能用这些旧队列继续提交并期待自动评分。**

这是实际核对队列配置后的结论，不只是因为比赛日期已过：

| 年份 | validation旧榜 | 当前保存的文件提交轮次截止（UTC） | 当前是否可继续提交 | 直接证据 |
|---|---|---|---|---|
| 2023 | 可打开并渲染旧成绩 | 2023-08-14 23:59 | 否，已关闭归档 | METS主页写明“Challenge Closed!”；教程写明“不再接受提交”；队列9615344无新轮次 |
| 2024 | 可打开并渲染旧成绩 | 2024-08-07 23:59 | 否，已关闭归档 | METS主页写明“Challenge Closed!”；队列9615545无新轮次 |
| 2025 | 可打开并渲染旧成绩 | 2025-08-07 23:59 | 当前否，重开未确认 | 队列9615901无新轮次；赛后管理员称重开时间尚未确定 |

所有截止时间来自此次完整读取的EvaluationRound列表，而非把论文/容器截止日期代作validation截止日期。每个METS文件队列当前列表均只有一条已结束轮次、没有下一页token。轮次是当前保存配置，不声称它重现了比赛期间每次临时调整的完整历史。

“不能继续提交”指当前普通参赛者经原validation队列提交的路径；不排除组织者日后重开、单独授权或线下代评。**本次没有实传文件验证后台，也没有证明评分容器已被删除或永远无法恢复。**

## 2. 每年入口及核查结果

### 2023

- [METS官方任务页](https://www.synapse.org/Synapse:syn51156910/wiki/622553)
- [validation Results](https://www.synapse.org/Synapse:syn51156910/wiki/622343)
- [文件提交教程](https://www.synapse.org/Synapse:syn51156910/wiki/622349)
- [Submission Dashboard](https://www.synapse.org/Synapse:syn51156910/wiki/622345)

任务页原文：

> Challenge Closed! … This challenge site is now archived. Data is no longer available for downloading here.

教程在Brain Metastases队列名称旁明确注明：

> no longer accepting submissions

旧Results仍有METS数据行，当前榜首可见提交ID为9739433，浏览器显示日期8/15/2023、Cases Evaluated为31。显示日期未强制当作UTC；不以晚于UTC截止的本地显示日期认定超期仍开放。

### 2024

- [METS官方任务页](https://www.synapse.org/Synapse:syn53708249/wiki/627504)
- [validation Results](https://www.synapse.org/Synapse:syn53708249/wiki/628815)
- [文件提交教程](https://www.synapse.org/Synapse:syn53708249/wiki/627754)
- [Submission Dashboard](https://www.synapse.org/Synapse:syn53708249/wiki/628814)

任务页同样有“Challenge Closed!”和“site is now archived”通知，导向2025的METS任务页。旧榜可渲染，当前第一条可见记录9748126，浏览器显示8/7/2024、Cases Evaluated为88。公开榜按Date倒序，但这只证明保留的历史成绩可读，不证明当前接受新提交。

### 2025

- [METS官方任务页（Task 4）](https://www.synapse.org/Synapse:syn64153130/wiki/631058)
- [validation Results](https://www.synapse.org/Synapse:syn64153130/wiki/632724)
- [文件提交教程](https://www.synapse.org/Synapse:syn64153130/wiki/632674)
- [Submission Dashboard](https://www.synapse.org/Synapse:syn64153130/wiki/632672)
- [赛后询问何时重开validation的论坛主题](https://www.synapse.org/Synapse:syn64153130/discussion/threadId=12508)

论坛主题是“when will the validation evaluation be available again please?”。浏览器显示管理员@astaraki于10/29/2025回复：

> The timeline for opening the validation queue submission has not been set yet but as soon as it is defined, we will announce it on the Synapse BraTS page.

这条回复不是2026年的新公告，所以单靠它不能断言今天仍关闭；本次另查当前9615901的完整轮次列表，仍只有2025-08-07结束的旧轮次，未出现新的开放轮次，两者共同支持当前判断。

旧榜实际渲染，当前榜首记录9756806，浏览器显示8/8/2025、Cases Evaluated为179，75条记录的第一页可见。2025榜仍注明7月9日以后只展示subject-wise DSC/NSD并对缺失预测施加惩罚，不能直接与本项目fixed103的病灶级DSC数值相减。

页面的“Last updated on 11/15/2026”是表格展示信息，**不是该METS模型提交/评分时间**，不能用它证明2026年旧validation仍出分。

## 3. 后台身份与轮次

精确队列ID来自官方Submission Dashboard中的widget，随后通过官方API核对队列name与contentSource：

| 年份 | 文件EvaluationID | contentSource | 当前轮次ID | roundStart（UTC） | roundEnd（UTC） | 当前轮次每日上限 |
|---|---|---|---|---|---|---|
| 2023 | 9615344 | syn51156910 | 230 | 2023-08-09 21:01 | 2023-08-14 23:59 | 2 |
| 2024 | 9615545 | syn53708249 | 305 | 2024-07-10 19:00 | 2024-08-07 23:59 | 2 |
| 2025 | 9615901 | syn64153130 | 395 | 2025-05-28 20:15 | 2025-08-07 23:59 | 1 |

2025通用教程写2次/日，但METS实际保存的轮次是1次/日；若重开，须重新核实时限与限制，不能沿用旧教程数字。

官方文档说明，EvaluationRound定义“a fixed time period during which submissions to an Evaluation queue are accepted”。查询使用：

```text
GET /repo/v1/evaluation/{evalId}
GET /repo/v1/evaluation/{evalId}/permissions
POST /repo/v1/evaluation/{evalId}/round/list  body: {}
```

最后一个虽使用HTTP POST，**仅是官方定义的列表读取API，不是上传或创建submission**。采集器只允许这一精确round/list路径使用POST，无任意写入端点。

文档：[Evaluation读取](https://rest-docs.synapse.org/rest/GET/evaluation/evalId.html)、[轮次列表读取](https://rest-docs.synapse.org/rest/POST/evaluation/evalId/round/list.html)。匿名当前权限均canView=true、canSubmit=false；匿名无提交权限本身不能证明认证用户也不能提交，因此关闭判断以完整过期轮次和明确官网通知为依据。

补充控制：`GET /evaluation?evaluationIds=…&activeOnly=true`和false都返回这三个已过期队列，与文档的activeOnly说明不一致，原因未进一步调查。**不把该列表返回当“队列开放”证据**，也不声称已查明服务端bug；采用精确round/list结果。

## 4. 持续评测入口的易混淆点

[2023 Continuous Evaluation: Adult Glioma](https://www.synapse.org/Synapse:syn51156910/wiki/622971)确实保留“Nov 14, 2023 - present”的持续评测介绍，但它属于**成人胶质瘤**，对应9615339，不是METS的9615344。本次没有验证该GLI入口今天是否仍有实际评分服务，不能推荐用METS预测上传到那里。

本次检索聚焦已有METS的2023–2025赛道，不把更早年份或其他疾病任务包装成可替代的脑转移瘤验证平台；也未对2026当前队列作新的状态查询。

## 5. 对我们横向比较的意义

1. 可向2025组织者申请：重开METS validation queue，或对同一官方validation预测做一次统一代评；同时询问评分版本、区域、NA/阴性/RC策略、提交限制及报告可否用于论文。这里只给出后续建议，没有替用户发帖或提交。
2. 我们目前四路成绩使用内部fixed103，不是官方无标签validation179；不能把103预测直接提交到要求官方病例集合的旧队列。
3. 即使换到旧官方validation，仍须先核对这些患者是否进入外部模型训练、选模、后处理调参或后续年度公开训练池。换年份和官方评分不能自动消除训练暴露。
4. 当前本项目K1、原E、2025镜像、Bio-Intel的统一MET成绩保持原记录；本次不重评分、解锁104、改模型或宣称官方排名。

## 6. 归档与缺口

- [结构化队列复核](AVAILABILITY_REVIEW.json)记录完整轮次、身份校验与未执行事项；[来源索引](SOURCE_INDEX.json)记录每份原件URL、获取UTC、HTTP状态、大小及SHA256。
- `sources/`保留官方wiki/队列/轮次；`docs/`保留API说明；`browser/`保留Chrome Playwright Bridge实际渲染节选，页面原件留本机，不当缓存发布。原创报告、工具、轻量队列JSON及来源身份进入Git。
- 补查2023 MLCube队列9615391匿名返回403，该辅助读取停止且失败响应/SHA保留，未登录绕过或重试，未把它归为队列关闭。其他容器队列未继续读取；主要结论只覆盖validation文件队列。
- 首次猜写API文档路径`GET/evaluation/id.html`返回AccessDenied；经官方索引定位正确`evalId.html`后读取成功。动态widget的Loading/login提示和浏览器console警告没有当作比赛关闭依据。
- 未登录核验个人认证/参赛资格；未实际上传；未验证评分runner存活；未联系组织者；未声称未来不可重开。完整官方测试集仍不可见，本次不获取MRI、GT、模型、镜像或数据包。

来源材料均为外部不可信参考，不作为本机/服务器操作指令。项目科学状态保持`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。
