# 3DGen 项目结构与高仰角视图转换管道

本文档梳理当前仓库中与“高仰角单图 -> 35°视图 -> 单图3D生成 -> 评测”相关的项目结构、模型设计、数据约定和调用关系。内容基于当前代码与 `group_view` 实验管道整理。

> 仓库中的大写 `README.md` 是上游 TRELLIS 模型卡；本文档描述本项目实际使用的组合管道。

## 1. 项目目标

当前实验比较五类输入视图对单图3D生成质量的影响：

- `HighElevation_0Roll`：高仰角、无 roll 原图；
- `HighElevation_RandomRoll`：高仰角、随机 roll 原图；
- `LowElevation_0Roll`：低仰角、无 roll 原图；
- `LowElevation_RandomRoll`：低仰角、随机 roll 原图；
- `Rotated`：将高仰角、无 roll 图像通过微调 Zero123 转换到 35°附近的结果。

核心问题是：虽然 35°已通过实验确定为较适合下游单图3D生成的视角，但当前 `Rotated` 输入 Hunyuan3D/TRELLIS 后的结果仍可能弱于直接输入高仰角原图。

## 2. 核心目录

```text
3DGen_new/
├── configs/
│   └── base.yaml                         # 角度估计器及其他训练配置
├── modules/
│   ├── angle_predictor.py                # 当前管道使用的仰角估计网络
│   ├── threestudio/
│   │   ├── configs/
│   │   │   ├── zero123-finetune.yaml     # Zero123 35°微调配置
│   │   │   └── zero123-generate.yaml     # Zero123 批量生成配置
│   │   └── custom/zero123_finetune/
│   │       ├── data.py                   # JSONL 配对数据读取与图像预处理
│   │       └── system.py                 # Zero123 微调、条件编码和验证生成
│   ├── Hunyuan3D-2.1/                    # Hunyuan3D 上游代码
│   └── trellis2/                         # TRELLIS.2 上游代码及 DINOv3 extractor
├── trainer/
│   └── trainer_elevation.py              # 仰角估计器训练/评估逻辑
├── tools/
│   ├── run_rotation_pipeline.py          # 仰角预测 + Zero123 转换 + 后处理
│   └── eval_docker/
│       ├── hunyuan_batch_to15_render.py  # Hunyuan3D 批量生成与渲染
│       ├── trellis_batch_to15_render.py  # TRELLIS 批量生成与渲染
│       ├── compare_glb.py                 # GLB 对齐、CD、PSNR等评测
│       ├── cd_psnr_by_pitch.py            # 按输入视角汇总和绘图
│       └── group_view/
│           ├── hunyuan.sh                 # Hunyuan group_view 入口
│           ├── trellis.sh                 # TRELLIS group_view 入口
│           ├── compare_glb_hunyuan.sh     # Hunyuan 评测入口
│           └── compare_glb_trellis.sh     # TRELLIS 评测入口
└── inference_elevation.py                 # 仰角预测离线评估脚本
```

实验数据位于仓库外部：

```text
/data/home/2024120101018/linzhuohang/3DGen/data/testset/group_view/
├── HighElevation_0Roll/
├── HighElevation_RandomRoll/
├── LowElevation_0Roll/
├── LowElevation_RandomRoll/
├── Rotated/
├── origin_glb/
├── hunyuan/
└── trellis/
```

## 3. 文件名与视角约定

输入图片和生成 GLB 通常使用：

```text
<object_id>_<yaw>_<elevation>_<roll>.<ext>
```

例如：

```text
4d55b54c4bd049beb74cc13a6230a50e_171_71_0.png
```

表示 yaw=171°、elevation=71°、roll=0°。

需要注意：当前 `Rotated` 图像内容约为 35°，但文件名仍保留源图高仰角。这不会直接改变 Hunyuan/TRELLIS 的生成条件，因为二者只接收图像；但会使 `compare_glb.py` 和 `cd_psnr_by_pitch.py` 将 Rotated 结果归入错误的 pitch 分桶。后续应分别记录：

```text
source_elev：原始高仰角
generated_elev：Zero123 输出目标角，当前为35°
```

## 4. 总体调用链

```text
HighElevation_0Roll/*.png
          |
          v
run_rotation_pipeline.py
  1. Angle Predictor 估计 ref_elev
  2. 构建 Zero123 JSONL manifest
  3. Zero123(ref_elev -> tgt_elev=35°)
  4. 截取预测栏、resize、meanshift
          |
          v
Rotated/*.png
     |                         |
     v                         v
Hunyuan3D                  TRELLIS.2
     |                         |
     v                         v
hunyuan/Rotated/*.glb      trellis/Rotated/*.glb
     |                         |
     +------------+------------+
                  v
           compare_glb.py
       mesh对齐 / CD / 渲染指标
                  |
                  v
       compare_glb_stage2_clip.py
                  |
                  v
        cd_psnr_by_pitch.py
```

## 5. 仰角估计模型

### 5.1 输入输出

输入为：

```text
[B, 3, 512, 512]，RGB，数值范围[0, 1]
```

输出为单个无约束标量，训练语义是角度制 elevation：

```text
[B]，单位为度
```

当前旋转管道使用权重：

```text
/data/home/2024120101018/linzhuohang/3DGen/data/angle_predictor.bin
```

### 5.2 网络结构

`modules/angle_predictor.py::ElevationRegressorNet` 包含两条特征路径：

1. 从零初始化的 ViT 主干：
   - 输入 512x512；
   - patch size 32，共 16x16 个 patch；
   - hidden dimension 768；
   - 14 层 Transformer Encoder；
   - 12 个 attention heads。
2. 冻结 DINOv3：
   - 图像做 ImageNet mean/std normalize；
   - 提取 DINO token；
   - 默认取第一个 token；
   - 通过 `dino_to_embed` 投影到 768 维；
   - 加到自建 ViT 的 CLS token。

最后使用两层 MLP 回归 elevation：

```text
Transformer CLS -> Linear(768,384) -> GELU -> Dropout -> Linear(384,1)
```

### 5.3 训练目标

`trainer/trainer_elevation.py` 中的训练损失为归一化角度 MSE：

```python
gt = gt_deg / 90
pred = pred_deg / 90
loss = MSE(pred, gt)
```

同时记录 degree-space MAE 和 RMSE。GT 被限制到配置的 `[angle_min, angle_max]`，模型输出本身没有 sigmoid/clamp 约束。

### 5.4 当前推理流程

`run_rotation_pipeline.py` 和 `inference_elevation.py` 会：

1. resize 输入到 512x512；
2. 单独提取归一化后的 DINO tokens；
3. 将未归一化的 `[0,1]` RGB 输入自建 ViT；
4. 将 DINO tokens 传入预测器；
5. 得到 `ref_elev`。

### 5.5 已确认风险

- 推理代码使用 `raw.abs().max() <= 1.5` 猜测输出是归一化值还是度数，但训练实现定义输出为度数。运行时猜单位可能将有效的 1°预测错误放大到 90°。
- checkpoint 使用 `strict=False` 加载，未报告 missing/unexpected keys；结构变化可能导致部分层保持随机初始化而不报错。
- 从零训练的 14 层 ViT 对标量回归偏重，可能学习类别和渲染域偏差。
- DINO 空间 tokens 被压缩成单个 token，顶部/侧面可见比例等空间信号利用不足。
- 输出没有限定在 0°~90°。
- 全局 MSE 与下游 `35-ref_elev` 的相机条件敏感性并不完全匹配。

### 5.6 建议改造

- 明确 checkpoint 输出单位，移除运行时自动判断；
- 加载时检查并打印 missing/unexpected keys，关键层缺失则失败；
- 使用 `90 * sigmoid(raw)` 等有界参数化；
- 改为冻结 DINO patch tokens + 轻量空间 attention head；
- 使用 SmoothL1 + elevation 分桶辅助分类；
- 重点报告 40°~75°任务域的分桶 MAE、bias 和 RMSE。

## 6. Zero123 图像扭转模型

### 6.1 数据格式

`zero123_finetune/data.py` 从 JSONL 读取配对：

```json
{
  "sample_id": "object_yaw_elev",
  "ref_img": "/path/to/reference.png",
  "tgt_img": "/path/to/target_35.png",
  "ref_yaw": 118,
  "ref_elev": 64,
  "tgt_yaw": 118,
  "tgt_elev": 35
}
```

参考图和目标图被转换为 256x256 RGB，并映射到 `[-1,1]`。RGBA 输入在白色背景上合成。

### 6.2 相机条件

`zero123_finetune/system.py` 构造四维相机条件：

```python
T = [
    radians(tgt_elev - ref_elev),
    sin(radians(tgt_yaw - ref_yaw)),
    cos(radians(tgt_yaw - ref_yaw)),
    radians(90 - ref_elev),
]
```

其中：

- `ref_elev` 是输入参考图的相机仰角；
- `tgt_elev` 是目标输出图的仰角，当前实验固定为35°；
- 第1维描述相对仰角变化；
- 第4维保留参考相机绝对仰角信息。

### 6.3 训练模块

基础模型来自 Stable Zero123 checkpoint。当前配置声明：

```yaml
train_unet: true
train_cc_projection: true
train_cond_stage: false
train_first_stage: false
```

但当前 `configure()` 实现实际只解冻 UNet：

```python
self.model.model.diffusion_model.parameters()
```

`train_cc_projection`、`train_cond_stage` 和 `train_first_stage` 配置尚未在实现中生效。因此负责融合 CLIP 图像特征与相机条件 `T` 的 `cc_projection` 实际仍被冻结。

训练损失沿用 Zero123 的 diffusion noise prediction loss，没有额外的身份保持、纹理一致性或轮廓一致性损失。

### 6.4 训练数据问题

当前 `train_35.jsonl` 的统计为：

```text
样本数：737,415
ref_elev范围：0°~90°
中位数：42°
ref_elev < 40°：约47.94%
```

这里统计的是 **Zero123 训练 manifest 中的配对记录及其 `ref_elev` 标注**，不是角度估计器训练集，也不是通过图像视觉或相机矩阵重新测得的物理视角。将这些记录描述为低仰角参考图，依赖数据集生成流程保证 `ref_elev` 与 `ref_img` 的真实渲染视角一致。

训练配置设置：

```yaml
ref_elev_min: 40
```

历史实现执行的是修改标签：

```python
ref_elev = max(ref_elev, 40)
```

而不是过滤样本。因此，约47.94%的 Zero123 训练配对原本标注为 `ref_elev < 40°`，进入数据加载器后其条件值会被改写为40°。在 manifest 标注准确的前提下，这会造成参考图真实视角与送入模型的相机条件不一致，并可能促使模型弱化对角度条件的响应。

当前代码已将 `ref_elev_min/ref_elev_max` 改为数据集初始化阶段的记录过滤条件，不再修改 manifest 中的 elevation 标签。使用 `ref_elev_min: 40` 时，标注低于40°的训练和验证配对会被排除，并在日志中报告过滤数量。

对于当前任务，更合理的策略是：

```text
过滤并只保留真实 ref_elev >= 40° 的训练对
```

而不是修改低仰角样本的角度标签。

### 6.5 验证和生成

验证阶段显式构造：

```text
CLIP reference embedding
+ camera condition T
+ reference VAE latent
```

再进行 DDIM 采样。当前默认/配置关注的参数包括：

```text
steps=50
guidance_scale=1
eta=1
```

对于保持同一物体身份的视角转换，`eta=1` 的随机性可能增加纹理漂移和结构幻觉。应优先测试 `eta=0`，并联合搜索 guidance scale。

### 6.6 建议改造

- 删除 `ref_elev_min` 标签 clamp，改成训练 manifest 过滤；
- 训练数据重点覆盖真实 `40°~75° -> 35°`；
- 增加少量 `35° -> 35°` identity 样本；
- 让 `train_cc_projection` 配置真正解冻该层；
- 显式打印每个可训练模块和参数量；
- 测试 DDIM `eta=0` 与不同 guidance；
- 引入低权重 DINO/CLIP identity consistency，避免全图 L1 约束不同视角像素；
- 分别评价视角准确性和对象身份保持，而不只看 diffusion loss。

## 7. 旋转管道

入口：

```bash
python tools/run_rotation_pipeline.py \
  --input_dir /path/to/HighElevation_0Roll \
  --output_dir /path/to/Rotated \
  --target_elev 35
```

### Step 1：估计输入仰角

`angle_predictor.bin` 对每张输入图预测 `ref_elev`。

### Step 2：生成 manifest

当前写入：

```json
{
  "ref_yaw": 0,
  "ref_elev": "predicted elevation",
  "tgt_yaw": 0,
  "tgt_elev": 35
}
```

由于 Zero123 条件使用 yaw 差值，`ref_yaw=tgt_yaw=0` 表达零 yaw 变化。需要保证训练和推理采用同一相机坐标约定。

### Step 3：Zero123 生成

管道调用 threestudio：

```text
launch.py --config configs/zero123-generate.yaml --validate
```

并用动态生成的 manifest 覆盖配置中的 `data.val_manifest`。manifest 显式包含 `tgt_elev=35`，因此优先于配置中的 `default_target_elev`。

### Step 4：后处理

验证保存图由以下三栏横向拼接：

```text
[reference | target | prediction]
```

当前后处理：

1. 取最右侧正方形作为 prediction；
2. resize 到1024x1024；
3. meanshift，默认 `sp=40, sr=13`；
4. 写入 `Rotated`。

强 meanshift 可能抹除细边、纹理和小零件。应保留原始 prediction，并做以下消融：

```text
raw prediction
resize only
light meanshift
current meanshift
```

## 8. Hunyuan3D 推理

入口：

```bash
bash tools/eval_docker/group_view/hunyuan.sh
```

该脚本读取五组输入，并显式输出到：

```text
$group_view_root/hunyuan/<input_group>/
```

默认预处理流程：

```text
PIL RGB
-> Hunyuan BackgroundRemover
-> Hunyuan shape pipeline
-> texture pipeline
-> GLB
```

Zero123 输出边缘可能比原图更模糊，二次背景移除可能进一步删除细结构或产生 alpha 光晕。因此需要对 `Rotated` 比较开启/关闭 Hunyuan preprocess 的结果。

`--target_elev` 在该批处理程序中主要用于生成后的展示渲染，不是 Hunyuan shape pipeline 的输入条件。

## 9. TRELLIS 推理

入口：

```bash
bash tools/eval_docker/group_view/trellis.sh
```

当前脚本已经显式设置：

```bash
--output_root $root/trellis
```

这保证生成目录与 `compare_glb_trellis.sh` 的读取目录一致。

默认预处理流程：

```text
PIL RGB
-> resize 1024x1024
-> pipe.preprocess_image
-> sparse structure sampler
-> shape structured-latent sampler
-> texture structured-latent sampler
-> GLB
```

所有组使用相同 seed=42 和采样参数。`--target_elev` 同样主要控制生成后的展示渲染，不作为 TRELLIS 图生3D条件。

## 10. GLB 评测

Hunyuan/TRELLIS 的比较脚本依次执行：

1. `compare_glb.py`
   - 按 object ID 找到 `origin_glb`；
   - 对生成 mesh 做尺度处理、yaw 搜索和 ICP 对齐；
   - 计算 Chamfer Distance；
   - 可生成固定/规范视图并计算 PSNR。
2. `compare_glb_stage2_clip.py`
   - 对规范视图计算 CLIP 一致性。
3. `cd_psnr_by_pitch.py`
   - 按来源目录、yaw、pitch、roll 汇总；
   - 输出折线、柱状图和热力图。

当前比较脚本使用：

```bash
--pitch 0
--skip_render
--skip_exist
```

`--pitch 0` 固定部分渲染视角；统计表中的原始 pitch 仍来自文件名。`--skip_exist` 会复用旧结果，做新一轮实验前必须确认输入图片、GLB和评测CSV的修改时间一致。

## 11. 已确认的问题清单

按优先级排列：

1. Zero123 `train_35.jsonl` 中约47.94%的训练配对标注为 `ref_elev < 40°`；历史实现会 clamp 条件，当前已改为过滤记录；
2. 配置中的 `train_cc_projection=true` 未在实现中生效；
3. TRELLIS 输出目录曾与评测目录不一致，现已修正；
4. `--skip_exist` 可能让新图片继续使用旧 GLB/旧评测结果；
5. Rotated 文件名保留源图高仰角，污染按 pitch 统计；
6. 角度估计 checkpoint 使用 `strict=False` 且不报告漏载；
7. 角度输出单位由数值范围自动猜测；
8. Zero123 后处理 meanshift 可能损伤下游所需细节；
9. Hunyuan 对 Zero123 图再次去背景，可能放大边缘问题；
10. Zero123 仅使用 diffusion loss，缺少显式身份保持目标。

## 12. 推荐实验顺序

### 阶段 A：先保证评测可信

1. 确认 TRELLIS 输出位于 `$root/trellis`；
2. 对新 Rotated 输入强制重新生成对应 GLB；
3. 强制重新生成比较 CSV/图表；
4. 修正 Rotated 的输出角度元数据或文件名；
5. 固定相同 object ID、seed、推理参数和样本集合。

### 阶段 B：图像后处理消融

保持 Zero123 checkpoint 和 target=35°不变，比较：

```text
raw prediction
resize only
light meanshift
current meanshift(sp=40,sr=13)
```

### 阶段 C：角度条件消融

保持目标35°不变，比较：

```text
文件名GT ref_elev
angle predictor ref_elev
错误/扰动 ref_elev
```

用于量化 Zero123 对相机条件的真实敏感性。

### 阶段 D：重新训练 Zero123

1. 只保留真实 `ref_elev >= 40°`；
2. 去掉 label clamp；
3. 真正解冻 `cc_projection`；
4. 加入 identity 样本；
5. 使用 `eta=0` 做确定性验证；
6. 分别评价视角准确性、身份保持和下游3D质量。

### 阶段 E：重构角度估计器

1. 验证当前 checkpoint 加载覆盖率；
2. 固定输出单位为度；
3. 建立 40°~75°分桶基线；
4. 替换为轻量 DINO spatial regression head；
5. 比较预测角度对最终 GLB 指标的传导影响。

## 13. 结果解释原则

完整管道包含两个连续的信息瓶颈：

```text
角度估计误差
-> 错误 Zero123 相机条件
-> 视角/身份生成误差
-> 背景与后处理误差
-> 单图3D重建误差
-> mesh对齐和评测误差
```

因此不能只通过最终 Chamfer/PSNR 判断某一个模型。每轮实验应至少保存：

- 原始输入图；
- angle predictor raw 输出和最终使用的 `ref_elev`；
- Zero123 原始 prediction；
- 后处理后的 Rotated 图；
- Hunyuan/TRELLIS 实际输入图；
- GLB 文件与生成日志；
- 对齐后的评测结果和对应配置。

这样才能把问题定位到角度估计、视图转换、图像预处理、3D模型或评测中的具体环节。
