# Xiaomi-Robotics-1 × Genie Sim 适配方案

目标：在 Genie Sim instruction 套件上 SFT XR-1(5B)，然后本地/远程闭环评测。
参照：π0.5 的 Genie Sim 适配(`Anonymous-694/ACoT-VLA@agibot_world_challenge`)。

训练需要 8 张 96 GB 级别的卡(实测在 8×H20 上)：5B 全参 + bf16 + FusedAdam 不含激活约
70 GB，48 GB 的单卡放不下。转换器、数据管线验证和本地评测在一张 48 GB 的卡上就够。

---

## 0. 数据集

**用 Genie Sim 当前发布的 `instruction/`(`instruction_and_robust.tar`，任务目录带 `_500` 后缀)。**

| 版本 | 用不用 | 识别方法 |
| --- | :---: | --- |
| 当前发布版 `instruction/` | ✅ | state 109 维 / action 38 维；`meta/.trimmed_to_md_v1`；原生分辨率(head 640×400，腕 1280×1056)，能用满 XR-1 的 16 万像素预算 |
| 旧版 `task_suite/`(ModelScope，2026-04) | ❌ | state 186 维、任务名不带 `_500`、夹爪量纲 0–120，episode 也不是同一批 |

> π0.5 的 `download_dataset.sh` 拉的是旧版 `task_suite/`，和它自己训练配置里的
> `instruction_and_robust/*_500` 对不上。这里不用它的下载脚本。

转换后的 JSON 只放元数据和数值，视频直接引用原始数据集里的 mp4(绝对路径)。

---

## 1. EEF：最关键的配置(已端到端验证)

XR-1 的 **state 是关节角，action 是末端位姿的相对增量**，增量在**当前末端自身坐标系**下表达
(`pos + Δ @ R.T`、`R @ aa2rotm(Δ)`)，夹爪也是相对量。Genie Sim 上的官方基线都发关节角，
所以 XR-1 是第一个走 EEF 通道的适配，这条通道必须自己验。

### 1.1 数据侧的字段(以 `info.json` 的 `field_descriptions` 为准)

| XR-1 字段 | 取自 | 坐标系 |
| --- | --- | --- |
| `proprios.{l,r}_arm_joint` | `state[30:37]` / `state[37:44]` | — |
| `proprios.{l,r}_gripper_pos` | `state[0]` / `state[1]` | — |
| `proprios.{l,r}_ee_pos` | `state[24:27]` / `state[27:30]`(`end/arm_position`) | arm_base_link |
| `proprios.{l,r}_ee_rotm` | `state[16:20]` / `state[20:24]`(`end/arm_orientation`，xyzw) → 3×3 | arm_base_link |
| `actions.{l,r}_ee_pos` | `action[2:5]` / `action[5:8]` | arm_base_link |
| `actions.{l,r}_ee_rotm` | `action[8:12]` / `action[12:16]`(xyzw) → 3×3 | arm_base_link |
| `actions.{l,r}_gripper_pos` | `action[0]` / `action[1]` | — |

**陷阱：`state/end/position` 是 base_link 系，`action/end/position` 是 arm_base_link 系，名字一样。**
proprio 必须取 `state/end/arm_*`，不能取 `state/end/*`。

### 1.2 验证结果

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| action EEF vs `state/end/arm_*`(lag 2) | 1.1 mm / 0.0065 | ✅ 同一坐标系 |
| action EEF vs `state/end/*` | **491 mm** / 1.27 | ❌ 配错的后果 |
| `state/end/arm_*` vs FK(state 关节) | **0.00 cm / 0.00°** | 末端 link、四元数顺序、关节顺序全部吻合 |
| action EEF vs FK(action 关节) | **0.000 mm / 0.0000°** | EEF 目标和关节目标是同一个目标 |

FK 用评测侧同一份 `G2_NO_GRIPPER.urdf`(`arm_base_link` → `arm_{l,r}_end_link`)。

### 1.3 评测侧

**观测**：`states.end_pose.arm_base_link.{left_arm,right_arm}` 由 corobotpolicy 用同一个 URDF
做 FK 得到(wxyz 转成 xyzw 后下发)。和数据集 `state/end/arm_*` 是同一个函数，逐位一致。

**输出**：默认在服务端解 IK、发 `JOINT_ABS`(见 §1.4)。也可以发 `EEF_ABS`：

```jsonc
"left_arm":  {"kind": "EEF_ABS", "values": [[x,y,z,qx,qy,qz,qw], ...], "base_link": "arm_base_link"},
"right_arm": {"kind": "EEF_ABS", "values": [[...], ...],              "base_link": "arm_base_link"},
"left_effector":  [[g], ...],   // 绝对值
"right_effector": [[g], ...]
```

- **`EEF_ABS` 必须显式写 `"base_link": "arm_base_link"`。** 默认是 `base_link`，不写的话仿真会再做一次变换，不报错
- 两条路径都**没有平滑**：`JOINT_ABS` 虽然传了 `smooth_alpha=0.5`，但 `filter_abs_joint()` 第一行就
  `return action`，后面是死代码(`utils/infer_post_process.py`)

### 1.4 IK 精度：仿真自带的不够，改在服务端解

仿真的 `EEF_ABS` 每步用当前实测关节作种子，调用 3 次限速的 `solve()`
(`ikfk_utils.eef_actions_to_joint`)。用仿真自己的求解器在 6,000 个数据集目标上实测
(`ik_check.py`)：中位 1.4 mm，但 **p90 15.7 mm、p99 25.8 mm**，约 10% 的步偏 1.5 cm 以上。

`g2_ik.py` 改在服务端解：用 IK-SDK 的精确 FK 做 Levenberg-Marquardt，沿动作块逐步串联，
不收敛时从实测关节重试，然后发 `JOINT_ABS`。同样 6,000 个目标：中位 0.013 mm、p99 0.26 mm。
本地闭环里，每个动作块最后一步的跟踪误差中位从 18.3 mm 降到 1.3 mm。数据里 EEF 目标 = FK(关节目标)，
所以两条路径在训练数据上等价。`--arm-output eef` 保留仿真 IK 作对照。

---

## 2. 其他字段

| | XR-1 期望 | 处理 |
| --- | --- | --- |
| 数据格式 | 每 episode 一个 JSON + 三路 mp4 | 转换器生成 JSON，**直接指向现有 mp4，不重编码** |
| 视频 | decord 读 | 原生版是 HEVC，decord 0.6 能解 |
| 图像尺寸 | `resize_image(factor=32, max_pixels=160000)` | 原生 head 640×400、腕 1280×1056 都会被缩到 16 万像素以内 |
| 动作步长 | 30 步 = 1.0 s @30fps | 默认 |
| 腰 | 相对增量 | instruction 里 action 腰 std = 0，state 有 ~1e-4 抖动；**两边都写 0**，评测不发腰 |
| 底盘速度 | 3 维 | 写 0 |
| 文本 | 多视角模板 + 任务描述 | 取 `annotations.json` 的 action_text，多步用 ", " 连接(和评测下发的字符串一致) |
| **训练帧** | 默认每帧都采 | **和 π0.5 一致：只在子任务段内采样**(见下) |

**训练帧和 π0.5 对齐。** π0.5 的 Genie Sim 配置都带 `dataloader_sampler="subtask"`
(`openpi/training/sampler.py`)：样本起点只取 `info.json` → `instruction_segments`
里的子任务区间(闭区间)，开头的静止段、最后一段之后的尾巴都不采。转换器把区间写进
每条 JSON 的 `sample_ranges`，`JsonDataset._sample_frames()` 按它采样
(字段缺失时退回每帧都采，官方数据不受影响)。轨迹本身完整保留，所以区间末尾的
动作块读的仍是真实的未来帧，和 openpi 一样。归一化统计量照 openpi，用全部帧。

| | 帧数 |
| --- | ---: |
| 全部 | 739,254 |
| 子任务段内(训练用) | **589,795(79.8%)**，各任务 74%–88% |
| 和 openpi `sample_subtask()` 逐帧对比 | **完全一致**(用桩数据集直接跑 openpi 原函数) |

照搬了 openpi 的一个怪处：指令里含 reset/return/default 且超过 90 帧的子任务截成 45 帧。
子串匹配把 straighten_object 的 "...returns it to the original position" 也算进去了，
500 条里截了 117 条。目标是和基线一致，所以保留。

---

## 3. 分步

### Step 1 · 转换器 + 数据管线验证(不训练)

1. `geniesim/lerobot_to_xr1.py`：LeRobot 任务目录 → 每 episode 一个 JSON
2. 转换器内置断言(配错直接挂掉，而不是静默通过)：
   - 每个 episode：`|action_pos[t] − proprio_pos[t+2]|` 中位数 < 1 cm
   - 四元数模长 ≈ 1
   - 视频存在、parquet 行数和 `episodes.jsonl` 一致
3. `tools/compute_normalize.py` 算统计量，`geniesim/make_data_config.py` 写进 `configs/data/*.yaml`
4. `geniesim/validate_xr1_data.py`：用 XR-1 的 `JsonDataset` 取样本，查图像形状、action `(30, 60)`、prompt
5. **把 action 用 `recover_action` 还原回绝对目标，和原始 `action[2:16]` 比** —— 往返误差应 ≈ 0。
   这一步同时验证了转换和推理侧还原用的是同一套约定

### Step 2 · 环境 + 数据

- `geniesim/setup_env.sh`；下载 `XiaomiRobotics/Xiaomi-Robotics-1-5B` 的 `model_states.pt`
- 全量 10 任务转换 + 重算统计量

### Step 3 · SFT

- 8×H20，DeepSpeed ZeRO-2；XR-1 默认学习率 `max_lr 2e-5`、warmup 500、cosine
- 启动：`bash geniesim/train_geniesim.sh <exp> [hydra overrides]`

**定稿配置**：30k 步，每卡 32 × 8 卡 = 全局 256，不累积；学习率用 XR-1 自带的；
`async_train=false`；每 10k 步存一次；子任务帧采样，straighten 的截断照搬 π0.5。
约 13 个 epoch，8×H20 上 54 小时。依据见 `PI05_VS_XR1.md` §5。

```bash
bash geniesim/train_geniesim.sh sft_instruction_30k \
    trainer.max_steps=30000 trainer.save_interval=10000 \
    model.params.model.async_train=false
```

输出在 `$OUT_ROOT/<exp>/`：`train.log`、`wandb/`，checkpoint 在
`project_xiaomi-robotics-1/<exp>/`(`epoch=0-step=10000.ckpt` … 加 `last.ckpt`，每个 64 GB)。

**冒烟实测(8×H20，每卡 batch 32，20 步)**

| 项 | 结果 |
| --- | --- |
| 退出 | 正常，20/20 步 |
| 步时 | 稳态约 6.4–7.2 s/步(见 `PI05_VS_XR1.md` §7)；首步 23 s，含 FusedAdam 现场编译 |
| loss | 5.26 → 3.09，全部有限(学习率还在 warmup 起点 5e-7) |
| **显存峰值** | **约 96 / 98 GB** |
| 序列长度 | 一个 batch 拼成一条序列，每样本约 535 token，32 个 = 17,120 |
| checkpoint | **每个 64 GB**(含优化器状态)；`save_top_k=-1` 全部保留 |
| 数据加载 | 单进程 263 ms/样本，8 worker/卡约 1 s/步，不是瓶颈 |

**每样本 token 数(实测，每任务 20 个样本)**

| | token |
| --- | ---: |
| 图像(头 480×288 → 135，两个腕 416×352 → 各 143) | 421(固定) |
| 其余：模板、指令、`<state>`、30 个动作 token、`<score>` | 100–138 |
| 其中语言指令 | 6–31(straighten_object 最长) |
| 合计 | 521–559 |

`CustomCollate` 把一个 batch 拼成一条序列，**超过 `MAX_LENGTH`(20,000)的样本会被静默丢掉**，
不会报错(`custom_collate.py` 里是 `continue`)。最长 559 × 32 = 17,888，每卡 32 放得下；
官方默认每卡 48 只能留 35 个，剩下 13 个直接被丢。

**每卡 32、不累积、8 卡 = 全局 256，和 π0.5 一样**(openpi 的 `batch_size` 是全局值，
平均分到所有卡，没有梯度累积)。显存余量小，训练时盯一下。

Lightning 默认每 50 步记一次日志；冒烟测试要加 `+trainer.log_every_n_steps=1`
才看得到 loss。`tools/train.py` 的 WandbLogger 改为读 `WANDB_DIR`(原来固定写到当前目录)。

### Step 4 · 评测服务

| 文件 | 作用 |
| --- | --- |
| `xr1_policy.py` | 本地评测的 websocket 服务，同时也是远程 agent 的后端(`build_backend(model_path, argv)`) |
| `g2_ik.py` | 服务端 IK(§1.4) |
| `replay_xr1.py` | 离线回放：按仿真客户端的方式组装请求，和数据集真实动作比；`--dump-payload` 导出 `check-inference` 用的 G2 payload |
| `ik_check.py` | 用仿真自己的求解器测 EEF_ABS 的 IK 精度 |
| `ik_sdk/` | genie_sim 的 G2 URDF 和求解器配置(MPL-2.0)，给没有 genie_sim 源码的机器用 |

```
入：images(三路，原生分辨率，不 letterbox —— 和训练一致)
    + states.arm_joint_states + gripper_states + end_pose.arm_base_link
    → compose_state(关节, 夹爪)
    → XR-1 推理 → (30, 60) 相对增量
    → recover_action(当前 end_pose 的 pos / rotm / 夹爪) → 绝对目标
    → 服务端 IK → JOINT_ABS(或 --arm-output eef：EEF_ABS + base_link=arm_base_link)
```

**已验证**：随机权重和预训练权重都跑通；官方 `check-inference` 通过；本地闭环首帧核对正常
(图像尺寸、夹爪约 -0.78、prompt)。10k checkpoint 在本地 907 条 episode 的 instruction 套件上：
端到端成功率 0.780，按步平均 0.863。

---

## 4. 一致性检查清单

- [x] 转换器断言全过(Step 1.2)
- [x] `recover_action` 往返误差 ≈ 0(Step 1.5)
- [x] 评测侧夹爪 `gripper_states` 和数据集 `state[0:2]` 量纲一致(约 [-0.79, 0])
- [x] 评测侧图像预处理和训练一致：两边都用原生分辨率 + 同一个 `resize_image`
- [x] IK 跟踪误差(§1.4)

## 5. 可选

- 静止帧裁剪：已由 π0.5 的子任务采样覆盖(§2)，不再单独做
- 历史帧：XR-1 默认单帧，不开 `history`
