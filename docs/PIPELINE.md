# GVHMR 项目 Pipeline 全景文档

> 面向二次开发的全局架构说明：模块划分、各模块职责、输入输出、数据流向。
> 基于 commit `ee960bb`（main 分支）代码梳理。

## 符号约定（shape 标注）

本文所有张量 shape 使用以下符号，与代码变量名一致：

| 符号 | 含义 | 取值/来源 |
|---|---|---|
| **`L`** | **视频长度，单位：帧**（Length） | 输入视频统一转码为 **30 fps**（`demo.py:89` `get_writer(..., fps=30)`），故 `L = 30 × 视频秒数`。例如 10.4s 的 tennis 视频 → L=312。对应 `get_video_lwh()` 返回的第一个值 |
| `B` | batch 大小（Batch） | demo 恒为 1（`DemoPL.predict` 自动加 batch 维）；训练时由 `data.loader_opts.train.batch_size` 决定 |
| `J` | 关节数（Joints） | 2D 观测用 COCO-17（`J=17`）；SMPLX body 部分为 21~22 关节（`body_pose` 维度 63 = 21×3） |
| `C` | 特征通道数（Channels） | 依上下文：图像特征 `C=1024`、相机角速度 `C=6`（旋转矩阵 6D 表示）、网络潜变量维度等 |
| `H`, `W` | 视频帧的高、宽（像素） | `get_video_lwh()` 的后两个返回值 |

注意区分：`L`（帧数）与 fps 的换算出现在 `demo.py:325`（`data_time = length / 30`）；时间维在所有模块中都是第 1 维（无 batch 时）或第 2 维（有 batch 时），如 `(L, 17, 3)` 与 `(B, L, 17, 3)`。

## 0. 一句话总览

**输入一段单人视频，输出该人物在世界坐标系（重力对齐、地面固定）中的 SMPLX 动作序列 + 相机坐标系中的动作序列。**

```
视频 (mp4)
  │
  ▼
[预处理] ──► bbox 追踪 + 2D 关键点 + 图像特征 + 相机运动
  │
  ▼
[GVHMR 模型] ──► SMPLX 参数（相机系 incam + 世界系 global）
  │
  ▼
[渲染] ──► 叠加视频 / 全局轨迹视频 / 结果文件 (.pt)
```

---

## 1. 顶层模块划分

| # | 模块 | 代码位置 | 职责 |
|---|------|---------|------|
| 1 | **预处理（Preprocess）** | `hmr4d/utils/preproc/` | 从视频中提取模型所需的所有观测信号 |
| 2 | **GVHMR 模型（核心）** | `hmr4d/model/gvhmr/` + `hmr4d/network/gvhmr/` | 观测 → SMPLX 动作参数（端到端 Transformer） |
| 3 | **参数编解码（EnDecoder）** | `hmr4d/model/gvhmr/utils/endecoder.py` | SMPLX 参数 ↔ 网络归一化特征空间 |
| 4 | **后处理（Post-process）** | `hmr4d/model/gvhmr/utils/postprocess.py` | 静止关节修正、IK 优化 |
| 5 | **渲染可视化（Render）** | `hmr4d/utils/vis/renderer.py` | SMPLX 网格渲染回视频 / 全局视角 |
| 6 | **训练/评测框架** | `tools/train.py` + `hmr4d/datamodule/` + `hmr4d/dataset/` | Lightning + Hydra 的训练与 benchmark |
| 7 | **配置系统** | `hmr4d/configs/` | Hydra 配置组合（demo / train / test） |

入口文件：

| 场景 | 入口 |
|---|---|
| 单视频 demo | `tools/demo/demo.py` |
| 文件夹批量 | `tools/demo/demo_folder.py` |
| 训练 / 评测 | `tools/train.py` |

---

## 2. 模块 1：预处理（Preprocess）

demo 中由 `run_preprocess(cfg)`（`tools/demo/demo.py:99`）驱动，4 个并行子模块，**结果均缓存到磁盘**（`outputs/demo/<name>/preprocess/*.pt`），二次运行自动复用。

### 1.1 Tracker —— 人体检测与追踪

- **代码**: `hmr4d/utils/preproc/tracker.py`（类 `Tracker`）
- **依赖权重**: `inputs/checkpoints/yolo/yolov8x.pt`
- **作用**: YOLOv8 逐帧检测人（class=0），ByteTrack 追踪，选出**最长的一条单人轨迹**；帧缺失时线性插值补齐，并做滑动平均平滑。
- **输入**: 视频路径
- **输出**: `bbx.pt` = `{"bbx_xyxy": (L,4), "bbx_xys": (L,3)}`
  - `bbx_xys` = [中心x, 中心y, 边长]，已按宽高比修正并放大 1.2 倍（`get_bbx_xys_from_xyxy`）

#### 📌 为什么用 `bbx_xys` 而不是 `bbx_xyxy`？（格式设计依据）

`bbx_xys` 不是"丢了宽高的 bbox"，而是**"以人为中心的正方形采样窗口"的参数化**——这是 HMR 领域（SPIN/CLIFF/HMR2.0 一系）的标准约定。下游所有环节只需要 3 个数：

| 下游用途 | 代码 | 为什么一个标量 `s` 就够 |
|---|---|---|
| 正方形裁剪 → 256×256 | `hmr4d/network/hmr2/utils/preproc.py:32` `crop_and_resize` | `center ± s·r/2` 构造正方形，仿射变换到模型输入，矩形框无用 |
| 关键点归一化到 [-1,1] | `hmr_cam.py:180` `normalize_kp2d` | 各向同性缩放，只需一个 scale |
| 深度估计 `gt_s = 2f/(tz·s)` | `hmr_cam.py:146` | 透视投影中"表观尺寸 ∝ 1/深度"，只需一个像素尺寸 |
| CLIFF 相机条件 | `compute_bbox_info_bedlam` | 中心 + 尺度即完整描述 |

相比 `xyxy` 的优势：尺度 `s` 显式化（直接参与深度公式）；取 `max(w,h)` + 平滑抗 YOLO 逐帧宽高比抖动；少一个冗余自由度。

**为什么 `max(w,h)` 之前还要做宽高比修正（0.75 = 192:256）？**
见 `hmr_cam.py:224-229`，修正**只放大不缩小**，且只在"框太宽"时改变 `s`：

- 框太窄（w < 0.75h，如站立）：加宽 w 到 0.75h，`s = max = h`，**不变**；
- 框太宽（w > 0.75h，如四肢展开/躺倒/遮挡截断）：加高 h 到 w/0.75，`s` 增大 33%。

其依据是"**长边可信、短边可疑**"的启发式：检测框短边方向可能被低估（脚部被遮挡截断、姿态导致比例异常），按人类标准比例（192:256，恰为 ViTPose 输入尺寸）把短边补齐，保证正方形窗口覆盖**预期中的完整人体**而非仅仅已检测到的像素。更关键的是**训练/推理一致性**：训练时 bbox 由 GT 2D 关节点 min/max 生成（紧贴身体、比例随姿态剧变），推理时来自 YOLO（有检测器 margin 习惯）——两者统一经过"0.75 修正 + 1.2 放大"后，`s` 才被规范化到同一个定义："按人类标准比例换算的完整身体范围 × 1.2"，保证深度估计和归一化在两侧统计分布对齐。

> ⚠️ **二次开发注意**：若更换检测器（如换成 mask 紧框的 SAM3），其 `s` 分布与 YOLO 松框不同，必须重新标定 `base_enlarge` 系数，否则引入系统性深度偏差。详见 `docs/proposals/sam3-tracker-replacement.md` §4。

### 1.2 VitPoseExtractor —— 2D 关键点估计

- **代码**: `hmr4d/utils/preproc/vitpose.py`（类 `VitPoseExtractor`）
- **依赖权重**: `inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth`（ViTPose-huge）
- **作用**: 按 bbox 裁剪人体区域（0.5 倍下采样 + 高斯模糊抗混叠），估计 **COCO-17 格式的 2D 关键点**；带 flip test 提精度。
- **输入**: 视频 + `bbx_xys`
- **输出**: `vitpose.pt` = `(L, 17, 3)` —— (x, y, confidence)

#### 📌 COCO-17 关键点定义

索引定义（左右以**被拍摄者自身视角**为准）：

| 索引 | 关键点 | 索引 | 关键点 |
|---|---|---|---|
| 0 | 鼻 nose | 9 | 左手腕 left_wrist |
| 1 | 左眼 left_eye | 10 | 右手腕 right_wrist |
| 2 | 右眼 right_eye | 11 | 左髋 left_hip |
| 3 | 左耳 left_ear | 12 | 右髋 right_hip |
| 4 | 右耳 right_ear | 13 | 左膝 left_knee |
| 5 | 左肩 left_shoulder | 14 | 右膝 right_knee |
| 6 | 右肩 right_shoulder | 15 | 左踝 left_ankle |
| 7 | 左肘 left_elbow | 16 | 右踝 right_ankle |
| 8 | 右肘 right_elbow | | |

- 输出张量最后一维为 `(x, y, confidence)`：x/y 是像素坐标，confidence 是热图峰值强度。
- 面部 5 点约束头部朝向，四肢 12 点覆盖全部运动链；**无手指/脚尖脚跟细节**——这是 SMPLX 手部与脚部姿态主要靠先验而非观测的原因之一。
- 左右交换对为 `[1,2],[3,4],...,[15,16]`，鼻子（0）是中轴点不参与交换（定义见 `hmr4d/utils/geo/flip_utils.py`）。

#### 📌 Flip Test 为什么能提精度

实现（`vitpose.py:38-41`）：原图与水平翻转图拼进 batch 维一次前向，翻转图的热图做"左右通道交换 + 空间翻转"还原，再与原热图平均：

```python
heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
heatmap = (heatmap + heatmap_flipped) * 0.5
```

原理（四层）：

1. **噪声平均（TTA ensemble）**：两次预测的随机定位误差近似独立，平均后方差减半，解码坐标更接近真值。
2. **强制施加人体双侧对称先验**（本质）：理想模型应满足水平翻转等变性（左右互换意义下），实际模型并不严格满足；flip test 在推理时把预测投影回等变流形，消除不对称自由度带来的误差。
3. **抵消训练数据偏差**：COCO 中右利手、拍摄角度等分布不均，模型对左/右侧关键点的精度不对称；翻转后偏差反向，平均抵消系统偏差。
4. **缓解裁剪边界效应**：人贴近 crop 边缘时卷积 padding 导致特征质量差；翻转后水平位置镜像，两次前向的边界伪影不相关，平均后减弱。

两点工程细节：

- **在热图层面平均而非关键点层面**：热图平均保留完整概率分布（置信度、多峰形态）；遮挡导致多峰预测时，坐标硬平均会落在两峰之间的错误位置，热图平均不会。
- **代价约等于零**：翻转图拼进 batch 维一次前向（`cat` → `chunk`），GPU 利用率高，实际耗时增加远小于 2 倍。

#### 📌 为什么需要 2D 关键点？（`obs` 条件的作用）

2D 关键点是四路条件信号中**唯一显式描述"身体各部位在哪"的一路**，是最直接的姿态观测证据。它经 `normalize_kp2d`（`hmr_cam.py:180`）变成 `obs` 后进网络，该函数做三件事：

1. **坐标归一化**：`2(x−center)/s` → bbox 局部坐标 [-1, 1]，姿态与位置/尺度解耦；
2. **置信度保留**：第三维保留 ViTPose confidence，网络自行学习"该信哪个关节"；
3. **出框即不可见**：落在 bbox 外的关键点置信度清零（`invisible_mask`），检测异常值自动屏蔽。

**为什么不能只靠图像特征 `f_imgseq`？** 两路是互补而非冗余：

| | `obs`（关键点） | `f_imgseq`（HMR2 特征） |
|---|---|---|
| 信息形式 | **显式几何**：关节精确 2D 位置 | **隐式外观**：纹理、轮廓、接触、场景上下文 |
| 对遮挡 | 低置信度明确标注"看不见" | 仍提供被遮挡肢体线索（衣着、惯性） |
| 域偏移 | 跨数据集稳定（点就是点） | 受 backbone 训练数据域影响 |
| 精度 | 像素级定位 | 全局、模糊 |

**训练视角**：`obs` 是"测量通道"——训练时由 GT SMPL 姿态投影 + 高斯噪声增广（`EnDecoder.get_noisyobs`，`noise_pose_k=10`）生成，使网络学到的映射天然匹配 ViTPose 带噪输出的分布；另有 `randomly_set_null_condition(p=0.1)` 以 10% 概率随机丢弃条件，强迫网络在观测缺失时退回动作先验——这是 ViTPose 偶尔整帧失败时 GVHMR 不崩的原因。

**系统设计视角**：GVHMR 把"人在画面中的状态"分解为三个正交通道——**姿态** → `obs`（位置无关）、**位置/尺度** → `bbx_xys`（`f_cliffcam` + `transl` 反投影）、**相机运动** → `cam_angvel`。关键点只承担第一个角色；若姿态观测混入位置与相机运动，网络将无法区分"位移来自人走还是相机晃"，GV 坐标系分解也就不成立。

> 推论：预处理环节真正的精度瓶颈在 **ViTPose 的关键点质量**，而非上游 YOLO 检测框（bbox 误差经平滑后影响很小）。改进 2D 姿态估计是提升整体精度性价比最高的方向。

### 1.3 Extractor —— 图像特征提取（HMR2.0 backbone）

- **代码**: `hmr4d/utils/preproc/vitfeat_extractor.py`（类 `Extractor`，内部用 `hmr4d/network/hmr2/`）
- **依赖权重**: `inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt`（HMR2.0a 的 ViT backbone）
- **作用**: 按 bbox 裁剪并 resize 到 256×256，提取每帧 **1024 维图像特征**。注意：**只用其视觉特征，不用其 SMPL 预测头**。
- **输入**: 视频 + `bbx_xys`
- **输出**: `vit_features.pt` = `(L, 1024)`

#### 📌 为什么用 HMR2 backbone？换通用特征（如 DINO）会有什么影响？

**结论**：架构上没有绑定，语义上深度绑定——**不重训直接换 backbone 必崩**，且重训后大概率打平或略差。当前决策：维持 HMR2 backbone，不做替换实验。

**架构耦合（很浅）**：网络对图像特征的唯一处理是 `LayerNorm(1024) → Linear(1024→latent)`（`relative_transformer.py:103-106`），`imgseq_dim=1024` 是配置参数，`Extractor` 类也只是"crop → backbone → (L, C)"。工程替换成本约半天。

**语义耦合（真正的障碍）**：`f_imgseq` 是条件信号，网络学到的是"HMR2 特征空间 → 动作"的映射，特征分布由 backbone 的训练目标决定：

- **HMR2 backbone**（ViT-H/16）：MAE 预训练 + **端到端为"像素→SMPL 回归"微调**，特征被姿态监督"掰"过，高度编码关节构型、身体朝向、形状等姿态判别信息；
- **DINO 等自监督特征**：从未被姿态监督，编码的是语义对应与部件关系，而非关节角度。

推理时直接换 = 让网络读一门没学过的语言。与 SAM3 换检测器的本质区别：`bbx` 的尺度漂移可用标量（enlarge 系数）标定修复；**1024 维特征空间的整体漂移没有低维校准手段**。

**重训后的预期**（假设性分析，未实验）：

- HMR2 占优：任务对齐特征在 HMR 领域几乎是公理（WHAM/GVHMR/TRAM 全系都用）；姿态判别信息"免费"存在，网络只需学时序整合；
- DINO 可能反超的场景：域偏移鲁棒性（奇装异服/动画风/极端光照）。注意 `obs`（2D 关键点）已承担显式几何，特征的独特贡献是**外观/上下文/遮挡线索**——恰是通用特征的强项，故差距可能没有直觉那么大。

**若未来真要验证，低成本的中间路线**（特征蒸馏适配器，无需重训）：

1. 任意无标注视频语料同时跑 HMR2 与候选 backbone，得到配对特征；
2. 训练小 MLP/线性层做 `候选特征 → HMR2 特征空间` 映射；
3. 推理时 `候选 backbone → adapter → 现有 GVHMR`。

适配器的可达精度本身即回答"两特征空间的姿态信息量差距"，可作为是否值得重训的决策依据（几小时 GPU vs 数天重训）。

### 1.4 相机运动估计 —— SimpleVO / DPVO（二选一或跳过）

- **代码**: `hmr4d/utils/preproc/relpose/simple_vo.py`（`SimpleVO`，默认）或 `hmr4d/utils/preproc/slam.py`（`SLAMModel`，DPVO）
- **依赖权重**: DPVO 才需要 `inputs/checkpoints/dpvo/dpvo.pth`
- **作用**: 估计逐帧**相机旋转**。SimpleVO 用 SIFT 特征匹配 + 几何法估计（间隔采样 step=8 再插值）；DPVO 是深度学习方法，更准但需编译 CUDA 扩展。
- **输入**: 视频（可选 `f_mm` 指定等效焦距，默认 24mm）
- **输出**: `slam_results.pt`
  - SimpleVO: `(L, 4, 4)` 相机位姿矩阵
  - DPVO: `(L, 7)` (tx,ty,tz,qx,qy,qz,qw)
- **跳过**: 命令行 `-s/--static_cam` → 不跑 VO，`R_w2c = I`

### 预处理产物汇总 → 模型输入

`load_data_dict(cfg)`（`demo.py:175`）把缓存文件装配成模型输入：

| 键 | shape | 来源 |
|---|---|---|
| `length` | int | 视频帧数 |
| `bbx_xys` | (L, 3) | Tracker |
| `kp2d` | (L, 17, 3) | VitPose |
| `K_fullimg` | (L, 3, 3) | 由 W/H 估计（或 `f_mm` 精确指定）的相机内参 |
| `cam_angvel` | (L, 3) | 由 VO 的 R_w2c 计算的**相机角速度**（`compute_cam_angvel`） |
| `f_imgseq` | (L, 1024) | HMR2.0 特征 |

---

## 3. 模块 2：GVHMR 模型（核心）

三层封装，从外到内：

```
DemoPL (hmr4d/model/gvhmr/gvhmr_pl_demo.py)          ← Lightning 封装，predict() 接口
  └─ Pipeline (hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py)  ← 条件组装 + 前后处理
       └─ NetworkEncoderRoPE (hmr4d/network/gvhmr/relative_transformer.py)  ← Transformer 网络
```

### 2.1 DemoPL —— 推理封装

- 加载 checkpoint：`inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`
- `predict(data, static_cam)`：自动加 batch 维；`normalize_kp2d` 把 2D 关键点归一化到 bbox 局部坐标 [-1, 1]，得到观测 `obs`；调用 Pipeline 并整理输出。

### 2.2 Pipeline —— 条件组装与解码

- **代码**: `hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py`
- **输入条件（4 路条件信号）**:
  | 条件 | shape | 含义 |
  |---|---|---|
  | `obs` | (B, L, 17, 3) | bbox 归一化的 2D 关键点 |
  | `f_cliffcam` | (B, L, 3) | bbox + 内参算出的 CLIFF 式相机参数（`compute_bbox_info_bedlam`） |
  | `f_cam_angvel` | (B, L, 6) | 归一化后的相机角速度（旋转矩阵6D表示） |
  | `f_imgseq` | (B, L, 1024) | 图像特征 |
- **输出**: `model_output = {"pred_x": (B,L,C), "pred_cam": ..., "static_conf_logits": (B,L,J)}`

#### 📌 什么是 CLIFF 相机参数（`f_cliffcam`）？

源自 CLIFF（ECCV 2022, *Carrying Location Information in Full Frames*）。它指出裁剪式 HMR 的根本缺陷：**crop + resize 抹掉了"人在完整画面中的位置"**。透视投影下，crop 里一模一样的两个人（同姿态同大小）在画面中心和画面边缘时，相对相机的射线方向完全不同——网络若不知道人在画面何处，就无法正确推断相机系下的 `global_orient` 与深度，画面边缘会产生系统性朝向误差。

实现：`compute_bbox_info_bedlam`（`hmr_cam.py:103-118`，BEDLAM 式实现）：

```python
bbox_info = [(cx - icx) / fl, (cy - icy) / fl, b / fl]   # (B, L, 3)
```

| 分量 | 几何含义 |
|---|---|
| `(cx − icx) / f` | 人到主点的水平偏移 ÷ 焦距 ≈ **水平方向角正切** tan θx |
| `(cy − icy) / f` | 垂直方向角正切 tan θy |
| `b / f` | 表观尺寸 ÷ 焦距 ∝ **1/深度** |

即"人在相机系中的方向 + 逆深度"的无量纲编码；焦距归一化使其不依赖具体相机（同位置同距离的人，任何焦距拍摄该值相同）。

在正交分解中的角色：`obs`（关键点）描述身体形状但不知人在画面何处，`f_cliffcam` 补上位置/尺度信息，使网络能：① 修正画面边缘的朝向预测；② 配合 `pred_cam` 经 `compute_transl_full_cam`（`hmr_cam.py:124`）把"crop 大小 + 画面位置 + 内参"换算成全透视相机系下的 `transl`。

#### 📌 相机角速度（`f_cam_angvel`）：告诉网络"画面动 ≠ 人动"

**定义**：逐帧相机旋转增量，由 VO 输出的相机旋转序列计算（`hmr4d/utils/geo_transform.py:567` `compute_cam_angvel`）：

```python
ΔR = R_w2c[t+1] @ R_w2c[t]ᵀ        # 相邻帧相机旋转差
cam_angvel = matrix_to_rotation_6d(ΔR)  # (F, 6)，末帧复制补齐
```

用旋转矩阵的 6D 表示（而非四元数/欧拉角）是因为它对网络更友好：连续、无符号歧义、无万向锁。进网络前还会用手工统计量做标准化（`gvhmr_pipeline.py:60-62`，`stats_compose.cam_angvel["manual"]`）。

**作用**：在正交分解中承担"相机运动"通道。GVHMR 要恢复**世界系**动作，就必须区分画面中的运动来自人还是来自相机：

- 相机左转时，画面中的人即使站着不动也会"向右漂移"——没有 `f_cam_angvel`，网络会把这种漂移误学成人体位移；
- 有了它，网络可以在预测 `global_orient_gv` 和 `local_transl_vel`（GV 系局部速度）时显式扣除相机自转的贡献，把人体运动还原到重力对齐的世界系。

**与全局重建的关系**：推理时 `get_smpl_params_w_Rt_v2` 用同一个 `cam_angvel` 把网络预测的 GV 系参数与相机运动复合，roll-out 出 `smpl_params_global`（见 §3 输出表）。即 `f_cam_angvel` 既是**网络输入条件**，又是**输出解码的几何桥梁**——相机运动信息只在预处理（VO）产生一次，被前向和反解两次消费。

**边界情形**：

- `-s/--static_cam`：`R_w2c = I` → `cam_angvel = 0`，网络走"相机静止"先验分支，后处理也换用 `pp_static_joint_cam`（见 §5）；
- VO 质量直接决定该信号质量：SimpleVO 只估旋转不估平移（平移对人体世界轨迹的影响由网络从图像证据中推断），DPVO 更准但更重——见 §1.4。

### 2.3 NetworkEncoderRoPE —— 去噪 Transformer

- **代码**: `hmr4d/network/gvhmr/relative_transformer.py`
- 结构：条件 embedding → 多层 Transformer（RoPE 相对位置编码）→ 回归头
- **注意**：名字叫 denoiser3d，但**推理是单次前向回归，不是扩散迭代采样**（训练时用 simple MSE loss，见 `gvhmr_pipeline.py:120+`）
- 额外预测 `static_conf_logits`：每个关节"该帧是否与地面相对静止"的置信度，供后处理使用

### 2.4 模型输出（`hmr4d_results.pt`）

经 EnDecoder 解码 + 几何换算后得到两套 SMPLX 参数：

| 键 | 内容 | 坐标系 |
|---|---|---|
| `smpl_params_incam` | `body_pose (L,63)`, `betas (L,10)`, `global_orient (L,3)`, `transl (L,3)` | **相机系**：`transl` 由 `pred_cam` 经 `compute_transl_full_cam` 反投影得到 |
| `smpl_params_global` | 同上 | **世界系**（gravity-view）：由 `global_orient_gv` + `local_transl_vel` 沿时间积分（`get_smpl_params_w_Rt_v2`）得到 |
| `K_fullimg` | (L,3,3) | 相机内参 |
| `net_outputs` | dict | 中间网络输出（调试用） |

**Gravity-View 坐标的核心思想**：网络直接预测重力对齐坐标系下的朝向和"局部速度"（每帧相对位移），再 roll-out 积分成全局轨迹——这就是论文标题中 "Gravity-View Coordinates" 的含义。

---

## 4. 模块 3：EnDecoder —— 参数编解码层

- **代码**: `hmr4d/model/gvhmr/utils/endecoder.py`（类 `EnDecoder`）
- **作用**: 在「SMPLX 物理参数空间」和「网络归一化特征空间」之间双向转换：
  - `encode(inputs)`：训练时把 GT SMPLX 参数编码为归一化 `target_x` (B, L, C)
  - `decode(pred_x)`：推理时把网络输出解码回 body_pose（r6d→axis-angle）、betas、global_orient、global_orient_gv、local_transl_vel
- 归一化统计量来自 AMASS+BEDLAM 数据集（`stats_compose`，配置 `endecoder: gvhmr/v1_amass_local_bedlam_cam`）
- 训练时还用 `get_noisyobs` 对姿态加高斯噪声做增强

---

## 5. 模块 4：后处理（Post-process）

- **代码**: `hmr4d/model/gvhmr/utils/postprocess.py`，推理时 `postproc=True` 触发
- 两个步骤：
  1. **静止关节修正**（`pp_static_joint` / `pp_static_joint_cam`）：利用网络预测的 `static_conf_logits`，找出"与地面静止"的关节（如站立时的脚），强制其世界轨迹平滑/固定，消除**脚底打滑（foot skating）**。静态相机时用 `_cam` 变体（利用相机静止先验）。
  2. **IK 优化**（`process_ik`）：对关键运动链做逆向运动学微调，使关节位置更贴合。

---

## 6. 模块 5：渲染（Render）

- **代码**: `tools/demo/demo.py:203-304` + `hmr4d/utils/vis/renderer.py`（基于 pytorch3d）
- 流程：SMPLX 参数 → `make_smplx("supermotion")` 前向得到网格顶点 → `smplx2smpl_sparse.pt` 稀疏矩阵转成 SMPL 网格 → pytorch3d 光栅化渲染
- 两个渲染器：
  | 函数 | 输出 | 内容 |
  |---|---|---|
  | `render_incam` | `1_incam.mp4` | 原视频帧上叠加人体网格（用 `smpl_params_incam` + 真实内参 K） |
  | `render_global` | `2_global.mp4` | 第三人称环绕视角：人挪到原点、面朝 -Z，带地面网格（用 `smpl_params_global`） |
- 最后 `merge_videos_horizontal` 横向拼接成 `<name>_3_incam_global_horiz.mp4`

---

## 7. 模块 6：训练 / 评测

- **入口**: `tools/train.py`（Hydra + PyTorch Lightning）
- **配置组合**:
  ```bash
  # 训练
  python tools/train.py exp=gvhmr/mixed/mixed
  # 评测（一次跑 3DPW/RICH/EMDB）
  python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=... ckpt_path=...
  ```
- **数据模块**: `hmr4d/datamodule/mocap_trainX_testY.py` —— "用 X 训练、用 Y 测试"的统一封装
- **数据集**: `hmr4d/dataset/`（训练：AMASS/BEDLAM/H36M；评测：3DPW/RICH/EMDB），均需预处理为 `inputs/<DATASET>/hmr4d_support/` 格式
- **训练 Pipeline 与推理的差异**: 训练时 `forward(train=True)` 走 loss 计算分支（simple MSE + 各类辅助 loss），且**不做后处理**，所以训练日志中的全局指标与测试脚本略有差异（README 有说明）

---

## 8. 模块 7：配置系统（Hydra）

- **目录**: `hmr4d/configs/`
  - `demo.yaml` —— demo 全部配置（含所有输出路径模板 `paths:`）
  - `train.yaml` —— 训练主配置
  - `store_gvhmr.py` —— 用 hydra-zen 注册组件（model/network/endecoder 等）
  - `exp/gvhmr/`、`global/task/` —— 实验与任务组合
- demo 的命令行参数通过 `compose(overrides=[...])` 注入（`demo.py:60-75`）

---

## 9. 外部依赖权重一览

| 权重 | 路径 | 用途 | 二次开发可替换性 |
|---|---|---|---|
| YOLOv8x | `inputs/checkpoints/yolo/` | 人体检测追踪 | 可换其它检测器（改 `Tracker`） |
| ViTPose-huge | `inputs/checkpoints/vitpose/` | 2D 关键点 | 可换 RTMPose 等（改 `VitPoseExtractor`，注意保持 COCO-17 格式） |
| HMR2.0a | `inputs/checkpoints/hmr2/` | 图像特征 backbone | 换 backbone 需重训 GVHMR（特征是模型输入条件） |
| GVHMR | `inputs/checkpoints/gvhmr/` | 核心模型 | — |
| DPVO | `inputs/checkpoints/dpvo/` | 可选 VO | 默认已被 SimpleVO 替代 |
| SMPL/SMPLX | `inputs/checkpoints/body_models/` | 人体模型 | 渲染用 SMPL，预测用 SMPLX |

---

## 10. 二次开发的常见切入点

| 需求 | 改动位置 |
|---|---|
| 换检测器/追踪器（多人、更稳） | `hmr4d/utils/preproc/tracker.py`（`get_one_track`） |
| 换 2D 姿态估计器 | `hmr4d/utils/preproc/vitpose.py` |
| 换 VO / 接入更好的 SLAM | `hmr4d/utils/preproc/relpose/` 或 `slam.py`；保证输出能转成 `R_w2c` |
| 导出结果到 Blender/Unity/机器人 | 读 `hmr4d_results.pt` 的 `smpl_params_global`（世界系，重力对齐）；参考 `hmr4d/utils/smplx_utils.py`、`hmr4d/utils/ik/` |
| 改网络结构/加条件信号 | `hmr4d/network/gvhmr/relative_transformer.py` + `gvhmr_pipeline.py` 的 `f_condition` + 重新训练 |
| 换人体模型/手部细节 | `hmr4d/utils/smplx_utils.py`（`make_smplx` 的注册表） |
| 自定义数据集训练 | `hmr4d/dataset/` 仿照现有数据集写，配置进 `hmr4d/configs/data/` |
| 批量处理自己的视频 | 直接用 `tools/demo/demo_folder.py` |

**开发提示**：预处理结果全部缓存在 `outputs/demo/<video>/preprocess/`，改模型后重跑不用重新预处理；删除对应 `.pt` 即可强制重算某一环节。`--verbose` 可额外输出 bbox/关键点的可视化叠加视频用于排查预处理问题。
