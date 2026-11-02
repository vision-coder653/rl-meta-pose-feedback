# Frozen Meta + 姿态反馈的肌电机械手控制实验

使用冻结的 Meta emg2pose tracking 模型提取肌电特征，以 PPO 训练固定机械手控制策略。此仓库只包含从头训练、评估、回放所需的源码和小型模型资源；不包含其他实验、专项续训脚本、原始数据、预训练权重、训练结果或缓存。

## 实验定义

- 输入为原始 2 kHz、16 通道 EMG。Meta CNN/TDS 和两层 LSTM 均冻结；取最上层 LSTM 的 **512 维 hidden state**。
- 50 Hz 输出控制命令，每次执行 10 个 2 ms 的 MuJoCo 物理步。
- Actor 使用 EMG hidden、机械手形态编码，以及由自身历史输出累积得到的命令姿态；不读取机械手实际关节位置或速度。初始人手姿态用于初始化 Meta tracking，段首机械手姿态通过参考姿态 retargeting 初始化。
- Critic 在训练中使用特权状态。Actor 输出关节增量，由机械手限位和动作掩码约束。
- 当前 reward 为几何跟踪代价的负数，没有张手姿态正则或独立的逐步碰撞惩罚；段首 retargeting 仍包含碰撞验收。
- 使用基于 KL 的动态学习率；训练集和验证集按用户分离，左右手均可参与。
- **这是离线窗口实验。** 编码器保留官方全窗口插值方式，不能据此视为已经实现因果实时流式解码；仓库不提供收敛或泛化成功的结论。

## 目录

```text
core/                              数据读取、运动学、仿真、retargeting、reward
configs/training_groups.json       手指对应关系和坐标定义
experiments/rl_meta_pose_feedback/  本实验网络、PPO、配置、入口和测试
resources/robot/                   固定机械手 MJCF、mesh 和 HOD 许可证
third_party/meta_reference/        官方等价性测试用的少量 Meta 源码
tools/                             数据准备和权重下载
```

`data/`、`assets/`、`runs/`、`results/`、`.cache/` 都是本地生成目录，被 Git 忽略。`SOURCE_MANIFEST.json` 记录发布前复制来源及原始哈希；发布时仅适配路径、依赖、入口范围和文档。

## 安装

已在 Windows、Python 3.12.7、CUDA 12.6 对应的 PyTorch 环境验证。以下命令均在仓库根目录执行；Linux 可将 `.venv/Scripts/python.exe` 换成 `.venv/bin/python`。其他系统/显卡组合尚未验证。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

直接运行源码入口即可，不需要 `pip install -e .`。配置与模型文件按仓库位置定位，请保留目录结构。

## 准备官方权重

```powershell
.venv/Scripts/python.exe tools/download_meta.py
```

此命令下载 Meta 官方 checkpoint 压缩包（约 248 MB），校验归档 SHA-256，再只提取 tracking_vemg2pose 权重到 `assets/meta/tracking_vemg2pose.ckpt` 并校验其 SHA-256。权重不会上传到本仓库。

如果已经有官方压缩包，可离线准备：

```powershell
.venv/Scripts/python.exe tools/download_meta.py --archive /path/to/emg2pose_model_checkpoints.tar.gz
```

## 准备数据

从 [Meta emg2pose](https://github.com/facebookresearch/emg2pose) 获取其数据并遵守原始使用条款。预期目录包含 `metadata.csv` 及该 CSV 引用的 HDF5 recording 文件；不要把原始数据提交到 Git。

可以放在 `data/emg2pose_data/`，也可以设置外部目录。PowerShell 示例：

```powershell
$env:EMG_DATA_DIR = '/path/to/emg2pose_data'
.venv/Scripts/python.exe tools/prepare_data.py --workers 4
```

此命令生成 `data/both_split.json` 和 `data/both_intervals.jsonl`。划分按用户进行，包含左右手；索引排除无效片段并检测时间戳间隔。每次训练、评估进程都需要相同的 `EMG_DATA_DIR`。索引记录文件大小和修改时间，移动或重新解压数据后应重新生成，命令不会覆盖已有索引；可用 `--output` 指定新目录并同步修改配置中的索引路径。

## 从头训练

先检查整个流程能否完成两个 PPO update：

```powershell
.venv/Scripts/python.exe experiments/rl_meta_pose_feedback/train.py --config experiments/rl_meta_pose_feedback/configs/ppo.json --output runs/smoke --updates 2
```

这只是运行检查，不用于判断模型是否学会。`ppo.json` 使用固定训练窗口池。正式训练可使用流式采样配置，持续抽取训练窗口：

```powershell
.venv/Scripts/python.exe experiments/rl_meta_pose_feedback/train.py --config experiments/rl_meta_pose_feedback/configs/ppo_raw_large.json --output runs/train
```

该配置 `total_updates=null`，按验证集早停规则结束；也可传 `--updates N` 明确限制更新次数。输出目录必须是新目录。CPU 可加 `--device cpu`。训练会在本地保存 `initial.pt`、`last.pt`、`best.pt`、指标和数据选择记录，均不纳入 Git。

## 评估和回放

```powershell
.venv/Scripts/python.exe experiments/rl_meta_pose_feedback/evaluate.py --checkpoint runs/train/best.pt --split val --output runs/train/validation
.venv/Scripts/python.exe experiments/rl_meta_pose_feedback/replay.py --evaluation runs/train/validation --episodes 0 --output runs/train/replay
```

打开生成的 `runs/train/replay/replay.html`。评估可使用 `--emg-control zero`、`--emg-control permuted` 或 `--zero-actions` 做输入替换和保持初始目标的对照；需要使用不同输出目录。单看 cost 变化不足以证明模型利用了 EMG 意图。

## 测试

安装依赖并准备 Meta 权重后，不需要真实数据集即可运行测试：

```powershell
.venv/Scripts/python.exe -m unittest discover -s experiments/rl_meta_pose_feedback/tests -q
.venv/Scripts/python.exe -m unittest discover -s tools/tests -q
```

测试覆盖官方模型等价性、冻结参数、PPO 更新、命令姿态反馈、双手数据划分、并行 retargeting、checkpoint 评估和发布准备工具。测试数据为本地生成的合成数据，产物写入临时目录。

发布前验证：21 项实验测试和 4 项准备工具测试通过；合成数据完成了索引生成、一次全新的 PPO update、独立评估及 HTML 回放。隔离 Python 进程确认项目模块全部来自此仓库。该验证只说明流程可执行，不代表真实数据上的训练效果；验证产物未发布。

## 来源与许可

Meta 源码/UmeTrack 模型及 HOD 资源的来源、固定版本和许可位置见 [THIRD_PARTY.md](THIRD_PARTY.md)。这些上游材料保留其各自许可，不应将整个仓库默认为 MIT 许可。项目代码的公开不表示附带任何未明确列出的许可授权。
