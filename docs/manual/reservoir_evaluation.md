# CoLM 水库评估

OpenBench 可直接读取 CoLM2024 的水库向量输出，复用站点评估流程。无需预先导出逐站文件或修改 DLB 的配置生成器。

| 评价项目 | CoLM fallback | ResOpsUS 观测 |
| --- | --- | --- |
| Dam_Inflow | qresv_in | inflow |
| Dam_Outflow | qresv_out | outflow |
| Dam_Water_Storage | volresv | storage |

## 输入与配置

原始模拟变量应为 `(time, reservoir)`，同一文件包含 `resv_GRAND_ID(reservoir)`。支持 `Month` 和 `single` 文件组织；Month 要求项目年份内每个月恰好一个匹配文件。各月水库顺序可不同。

```yaml
project:
  name: reservoir_inflow
  output_dir: ./output
  years: [2001, 2002]
  tim_res: Day
  grid_res: 0.1
  num_cores: 1
  station_plots: false

evaluation:
  variables: [Dam_Inflow]

reference:
  Dam_Inflow: ResOpsUS
  overrides:
    ResOpsUS:
      root_dir: /data/ResOpsUS
      fulllist: /data/ResOpsUS/list/prepared.csv

simulation:
  D:
    model: CoLM2024
    root_dir: /data/D/history
    data_type: grid
    tim_res: Day
    data_groupby: Month
    prefix: case_hist_unitcat_

metrics: [KGE, RMSE, correlation]
scores: []
comparison:
  enabled: false
statistics:
  enabled: false
```

例如源文件为 `case_hist_unitcat_2001-01.nc`。保留 `data_type: grid`：OpenBench 在解析实际变量和文件结构后，仅将水库向量的运行时类型转为 `stn`，不写回用户 YAML。普通经纬度网格继续使用原流程，已转换的逐站 `stn` 配置继续可用。

CoLM2024 内置三个项目的维度和 ID 字段。其他字段名可在模拟变量级覆盖：

```yaml
variables:
  Dam_Inflow:
    varname: qresv_in
    station_dim: reservoir
    station_id_var: resv_GRAND_ID
```

维度/ID 字段也可放在模型 registry 的变量映射中。`station_dim` 和 `station_id_var` 是非空字符串。首版原生多文件水库路径针对上述三个项目及 ResOpsUS 参考源；这些字段不会新建一种 `data_type`。

## 参考清单

CSV 必须有 `ID,SYEAR,EYEAR,LON,LAT,DIR`；`Flag` 可选，缺省启用，接受 true/false 或 1/0。每行指向一个观测文件，包含该评价项目对应的科学变量。相对 DIR 先按清单目录解析，再按参考根目录及原有 basename 规则解析。建议使用可移植的来源根目录相对路径或绝对路径。

显式 fulllist 优先使用，不要求 0.1°/0.25° 分辨率。未指定时兼容原来的分辨率清单。数字 ID 与 `ResOpsUS_<数字>` 统一为 GRanD ID；归一化后的重复 ID 明确报错，需要先整理清单，不自动选择匹配坐标。

没有显式参考年份时，从清单推导覆盖包络；显式 `years` 仍限制评估窗口。各水库使用自己的观测年份，包络不保证整个期间都有有效数据。原始观测 NetCDF 不被修改。

## 匹配、时间和单位

共同水库取区域/年份内启用观测与所有选定模拟、所有文件中 ID 的交集。所有模拟采用同一集合；某模拟全缺测的水库统一排除。没有共同有效水库时失败。分辨率匹配坐标不作为模拟水库身份，不能用下标或空间最近邻替代 GRanD ID。

每个文件先按 ID 选取，再拼接时间；沿用模型基于真实文件名的时间偏移规则，之后交给标准站点日尺度重采样和单位处理。流量支持 m³/s，储量支持 m³ 或 mcm；同一源的文件间单位必须一致，未知单位直接报错。储量参考的 million cubic meters 与模拟 m³ 通过现有单位系统转换，RMSE 单位以产物属性为准。

CF 填充值和 CoLM 的 -1e36 作为缺测，不填零。缺月、无效/重复 ID、缺变量、混合网格和水库结构、重复/非单调时间或处理期间源文件变化均报错。若 CTL 不输出水库变量，必须由用户明确选择适合评估的模拟，OpenBench 不自动移除 CTL。

## 检查与产物

先执行 `openbench check config.yaml`。该命令检查水库布局、月份及 ID 元数据，不读取水库时序数组；通过后再运行 `openbench run config.yaml`。预检查不等同于科学评估成功。

使用已有的站点 NetCDF、指标 CSV 和项目清单。`reservoir_matching/<项目>_<参考源>_<模拟>.json` 记录源文件签名、共同/选定 ID、未匹配观测和全缺测水库。匹配集合及全部模拟源签名参与缓存判定；仅重绘模式从运行记录和项目站点清单恢复，不要求原始数据仍在线。

原 DLB 外部转换脚本可继续使用。新接入不会补足观测覆盖；例如 ResOpsUS 没有 Amazon 区域观测时，仍无法得到该区域水库观测评估。
