# H3 本地依赖环境

本机为 RTX 5060 8GB、32GB 内存。本文区分依赖验收、真实模型试跑和成片质量；本地执行结果见末节，不把节点注册通过等同于生成通过。

## 启动与持久化

`docker-compose.override.yml` 自动叠加到原配置，使用 `comfyui-sos-comfyui:h3-quality`。五个数据挂载、GPU 设置、`--disable-cuda-malloc` 和 1.5GB 显存预留保持不变，管理器入口仍启用。当前完整 LoRA 修复及验证边界见末节。

```bash
docker compose config --quiet
docker compose up -d --no-deps --no-build comfyui
```

重建派生镜像需要本机 `temp/h3-build/wheels/` 中的 SageAttention 与 soundfile 两个固定 wheel，权重和 wheel 不提交 Git。构建前同步受版本管理的缺包锁文件到小构建上下文：

```bash
cp h3_local_requirements.txt temp/h3-build/wheels/h3_local_requirements.txt
cp h3_legacy_requirements.txt temp/h3-build/wheels/h3_legacy_requirements.txt
cp h3_fused_bypass.patch temp/h3-build/wheels/h3_fused_bypass.patch
docker build -f Dockerfile.h3-local -t comfyui-sos-comfyui:h3-quality temp/h3-build/wheels
```

`Dockerfile.h3-local` 校验 wheel 摘要并使用 `--no-deps` 安装，禁止顺带升级或降级 PyTorch。Fill 还会整体导入其音频、SDK、PDF 等节点，因此将独立环境实际验证所需的 19 项缺包以版本和 wheel 摘要固定在 `h3_local_requirements.txt`，安装启用 `--require-hashes`，不全量安装扩展的浮动 requirements。它基于本机已有、验收过的 `vision-fixed` 镜像，不是适用于任意设备的通用安装包。

## 固定版本

| 组件 | 本次版本或提交 |
| --- | --- |
| PyTorch / CUDA | `2.11.0+cu128`，未修改 |
| ComfyUI | `0.33.0`，仅回移植官方融合/旁路 LoRA 修复，见末节 |
| 管理器 | 内置 `comfyui_manager==4.2.2`，不另装第二份 |
| T8 采样扩展 | 原作者 v1.87.0：`T8mars/comfyui-minimax-h3-audio-T8@4af2a0256eb5dccc0dabc6256de6b9527bfd8531` |
| SageAttention | 官方 `thu-ml/SageAttention@d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`，版本 `2.2.0` |
| Sage wheel SHA256 | `93ecec1aeee32f4f58f7d9b908f8b442b8dea678f9f5ab935ae3bc288d0658bb` |
| Fill 音频导入依赖 | `soundfile==0.14.0`，以固定 wheel 和 `--no-deps` 补齐 |
| Derfuu 旧转换节点 | `Derfuu/Derfuu_ComfyUI_ModdedNodes@d0905bed31249f2bd0814c67585cf4fe3c77c015`，v1.0.1 |
| Impact Pack 旧开关节点 | `ltdrdata/ComfyUI-Impact-Pack@429d0159ad429e64d2b3916e6e7be9c22d025c3c`，v8.28.3 |
| Impact 最小新增依赖 | `piexif==1.1.3`、`dill==0.4.1`，固定 wheel 摘要，不解析传递依赖 |

管理器配置保持 `security_level=normal`、安全扫描及 SSL 校验，下载渠道改为 `https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main`。将官方支持的 `use_unified_resolver` 设为 `False`，避免每次重启都无条件重算并安装全部旧扩展依赖；界面内明确发起的安装仍可使用正常逐扩展流程。原 `custom_nodes/ComfyUI-Manager` 目录不删除，内置模式会阻止重复加载。

启动配置使用 `--enable-manager` 启用新版管理器界面，不再附加 `--enable-manager-legacy-ui`。管理器仍为上述固定 Python 包版本，不重复安装或更新依赖；后续主存调优见末节。

T8 与 Sage 原作未达一万星，但目标工作流需要其专用接口，没有已核验的一万星等价替代；采用固定原作，不伪造同名节点或替换成不等价 LoRA。

T8 最新版的附加保存节点依赖 `Hidden.execution_list`，当前运行镜像没有此新 API。固定 v1.87.0，而非接受部分注册或修改核心；已确认其 `h3_t8/nodes.py`、`sampling.py` 和 `core.py` 与最新版逐字一致，目标采样器和 H3 核心计算不变。

v1.87.0 全部 373 个节点的结构定义在实际镜像下通过检查。独立真实 ComfyUI 服务同时加载 Fill 与 T8，拼图节点和双时钟采样器均在 `/object_info` 中，本轮无扩展导入失败。

## 模型清单

四个主权重来自公开仓库 `Comfy-Org/MiniMax-H3@e5eb578a89295337b8ff433a035929ce0279e0b6`：

- `models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors`
- `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `models/vae/minimax_h3_video_vae_fp16.safetensors`
- `models/vae/minimax_h3_audio_vae_fp32.safetensors`

另外两个专用权重：

- `models/loras/minimax_h3_turbo_4步加速_comfyui_T8.safetensors`，来自 `RunningHubAI/rh-minimax-h3-turbo-4-comfyui-t8-lora@77c381d0670b1cfbc1f98e1a5eab99689e3189da`。
- `models/RMBG/BiRefNet/BiRefNet_toonout.safetensors`，来自 `1038lab/BiRefNet@4d000788a9698c7f8d67c8c6ce2b40c768f5b909`。

合计 57,203,835,765 字节。下载保留断点，完成后逐文件核对固定发布摘要、大小和 safetensors 头；未完成文件不能作为可用模型。

2026-10-21 依赖验收阶段：上述 6 个文件已全部完成大小、SHA256 及安全张量头校验，该阶段仅校验文件；后续真实 H3 执行结果见末节。

## 构建和验收边界

构建发生在独立容器：GCC 12、NVIDIA 官方 CUDA 12.8.93 编译器、`TORCH_CUDA_ARCH_LIST=12.0`、单编译任务。生产容器不安装编译器，不修改原作源码。

Debian 13 的 glibc 2.41 与 CUDA 12.8 头文件有声明冲突，复用 [Sunshine 固定版本的兼容补丁](https://github.com/LizardByte/Sunshine/blob/3494b23e20b13d27b88fa08b97d07cb6d917f919/packaging/linux/patches/x86_64/cuda-12-math_functions.patch)。只作用于临时工具链，断言恰好修改 12 条声明，不改生产系统头文件、ComfyUI 或注意力算法。补丁 SHA256 为 `a3a0e5c1f608b18b1269fea51ac34ca555ad4dba32d8c32877831271602a2962`。

候选镜像在 RTX 5060 上通过 48 项小张量测试：FP16/BF16、1024/1031 长度、两个随机种子、连续/融合 QKV 非连续布局，以及 KJ 实际 H3 注意力函数的分头路由。相对 FP32 SDPA 的最大相对 L2 误差为 0.037675，最小余弦相似度为 0.999290，测试峰值显存为 222,526,464 字节。

测试采用 eager（即时执行），不使用 CUDA Graph；未加载 H3 模型，也未验证真实权重、RoPE、长视频显存峰值或成片质量。当前 CUDA 12.8 对 Comfy Kitchen 优化内核的限制仍存在，不能把注意力验收理解为整个生成链已通过。

本机下载、构建、验收脚本及完整结果保存在 `/opt/maoning/Reference/h3-local-setup-20261021/`；它们不进入 ComfyUI 核心或 Git 中的模型目录。

H3 主路径依赖验收镜像为 `50460a79a4f1f7552432ba26b4f4675c65c89fba33c31bd9a6743f84e6522e73`，补齐旧分支前另保留为 `comfyui-sos-comfyui:before-legacy-20261021`。该阶段重启后扩展导入失败数为 0，T8 实际注册 373 个节点；H3 模型、编码器、VAE、LoRA 和 toonout 均出现在相应选项中，所有活跃后端节点齐全。管理器版本接口与临时 Chrome 页面检查通过，48 项 GPU 验收在该镜像再次通过。

与原基础镜像比对，原有 241 个 Python 包无版本变更或移除；新增 21 项固定依赖。另外既有扩展自身启动时补入 8 项小依赖（Socket.IO 相关 5 项及 jsonpointer、jsonpatch、lark），没有变更已有包。关闭的是管理器的无条件批量重算，不是修改这些扩展的自有启动行为。

补齐依赖阶段原工作流保持不变，SHA256 为 `ea78e0e23d665a31f34e4c762a84524248b20b846df9dfeb3b138621335baa41`。其 11 个 `DF_Int_to_Float` 和一个 `ImpactSwitch` 全处于旁路且没有连向输出，不是 H3 主路径依赖；随后用户明确要求保留旧分支并尽量补齐全部缺失节点，因此按下节安装原作，该阶段未修改画布或伪造同名节点。8 个空参考槽、节点 231/268 引用的本机不存在图片，仍需在真正试跑前整理；依赖齐全不表示原图可以不改素材直接排队。

## 补齐旧分支

两个原作均低于一万星，Derfuu 在 2024-11 归档；作为用户明确选择的旧图兼容例外，不将其当作活跃的新项目选型。两者源码直接从原作者 GitHub 克隆并固定上表提交，不修改原作实现。

下列命令仅适用于两个目标目录尚不存在的安装环境；已有目录应先检查本地改动，不覆盖：

```bash
git clone --filter=blob:none https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git custom_nodes/Derfuu_ComfyUI_ModdedNodes
git -C custom_nodes/Derfuu_ComfyUI_ModdedNodes switch --detach d0905bed31249f2bd0814c67585cf4fe3c77c015
git clone --filter=blob:none https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/ComfyUI-Impact-Pack
git -C custom_nodes/ComfyUI-Impact-Pack switch --detach 429d0159ad429e64d2b3916e6e7be9c22d025c3c
```

Derfuu 使用现有 NumPy 和 ComfyUI，自身没有安装脚本或新依赖。Impact 完整包硬导入所需的 segment-anything、scikit-image、SciPy、OpenCV、Transformers 等已存在，镜像仅通过 `h3_legacy_requirements.txt` 补充 piexif 和 dill；不再装一份会覆盖现有 OpenCV 的 headless 发行包，不安装整份无上限 requirements。

本次不运行 Impact 的 `install.py`，避免其自动下载 SAM ViT-B、生成配置及清理旧目录；正常原作导入无需此脚本。SAM2 在源码中明确为可选依赖，这次未安装，也未下载其权重。开关节点验收通过不等于 SAM2 分割功能已验收。

独立真实 CPU ComfyUI 注册 Derfuu 全部 31 个、Impact 全部 197 个节点。通过 `/prompt` 实际执行 17 项无模型图：15 个整数值转 FLOAT（含旧图 5–15）及两路 lazy 开关，返回数值、分支标签和索引均符合预期。过程未执行 H3、未生成视频、未写用户输入或模型。检查脚本、安装前基线和结果位于 `/opt/maoning/Reference/h3-legacy-setup-20261021/`。

2026-10-21 正式服务复验：当前镜像 `a9796897817f2ef121062bc9dba942e18135a163fe17a1833ee842f7b67297cf`，共注册 3874 个节点，原 3646 个无移除。目标工作流 104 个节点涉及的 49 种后端类型全部可用，包含旁路分支，缺失数为 0；前端虚拟节点不计入后端。原 270 个 Python 包版本全部未变，仅新增上述两个包。17 项实际执行再次全通过，工作流摘要不变，T8 373 个节点、六个模型文件和选项、管理器版本接口、两原作前端脚本均通过检查。重启日志没有扩展导入失败或这些测试的执行错误；不表示其他扩展可选功能或完整 H3 生成已通过测试。

## 恢复原服务

仅在队列为空时操作。显式指定原 Compose 文件即可不叠加 H3 配置：

```bash
docker compose -f docker-compose.yml up -d --no-deps --no-build comfyui
```

原镜像保留为 `comfyui-sos-comfyui:vision-fixed` 和 `comfyui-sos-comfyui:before-h3-20261021`，镜像 ID 为 `ba67d5aec57caf246473bc7c41ac05ad647e903775d2f05f057aac36fd489277`。管理器原配置备份在 `user/__manager/config.ini.before-h3-20261021.bak`，恢复前先检查是否已有用户后续修改。

## 视频输入与真实 H3 试跑（2026-10-21）

本机 `user/default/workflows/minimax/H3+SpriteSheet+outline+fix.json` 的视频输入已换为原作 `VHS_LoadVideo`：保留 251/254/256，移除 252/253/255 拆分器，图像接输出 0，音频接输出 2，固定 24fps、最多 360 帧。原三路旁路状态、其他生成参数和提示词保留；空槽不是自动判断，启用前必须选择有效文件。静音视频应断开该路 AUDIO，避免惰性音频读取报错。

真实原生载入发现并修正两项旧画布兼容问题：RMBG 新增 `sensitivity` 后，旧控件序列发生错位，现补默认值 1，保留原模糊 0、偏移 -1 等含义；旧 ImpactSwitch 的 `select/sel_mode` 移到前列，按输入名迁移 11 条边的目标槽位，保留其选择 6、全部旧节点和旁路状态。没有修改两个原作扩展的源码。

原图逐字节备份及原启动配置位于 `user/default/backups/h3-local-trial-20261021/`。原图的八个空图片和两个缺失图片引用仍保留，不擅自替换用户素材。可重跑的成功副本另存于同一 workflows/minimax 目录：`H3_SpriteSheet_8GB_trial_20261021.json` 是小样；`H3_SpriteSheet_8GB_original_size_ffn2_trial_20261021.json` 是原输出尺寸成功配置。未加分块的 `H3_SpriteSheet_8GB_original_size_trial_20261021.json` 为失败对照，不推荐直接排队。试跑使用本机已有 Hero 图及含声视频，参考视频仅读取 256×256、48 帧，停用其他空参考；明确不是原缺失素材的质量复现，也未验证三路高分辨率长参考同时启用。

实际运行镜像支持并已启用：

```text
--disable-cuda-malloc --reserve-vram 1.5
--async-offload 1 --cache-ram 12 12
--disable-cuda-graphs
```

动态 VRAM 在当前 NVIDIA/PyTorch 环境默认启用，日志已确认。当前不启用 `--fast-disk` 或 `--disable-pinned-memory`，允许原生主存权重驻留与锁页传输。`--cache-ram 12 12` 的两个值是 GiB 可用内存压力阈值，不是缓存容量上限：当前新增锁页路径通常检查约 6GiB 余量，节点间尝试恢复约 12GiB。它们不是硬预留，仍可能换页或主存不足；本机实测边界见末节，不使用无预算的 `--high-ram`。

下述首次试跑及保守回退模式使用 `--fast-disk --async-offload 1 --disable-pinned-memory --cache-none --disable-cuda-graphs`。fast-disk 从原权重 mmap/偏移流式读取，不新增整模型交换文件；cache-none 禁用节点结果/对象的持久复用缓存，不代表所有内存都能立即释放。当前镜像中 `--lowvram` 在动态模式不起作用，`--novram` 会关闭动态模式，因此不堆叠这些参数；不全局强制 FP16，不修改核心或采样算法。

未增加 Easy Use 清理节点：其 `cleanGpuUsed` 清自有缓存并调用 `unload_all_models()`，不能减少 H3 活跃采样张量峰值。保留原 KJ Memory Efficient Sage Attention 和 LoRA 链，不叠加另一个可能被已有 attention owner 覆盖的注意力补丁；实际原尺寸 FFN 溢出后，只增加原作 T8 的 FFN 分块。

低负载完整图已实跑：256×160、22 帧、24fps、8 步，任务 `ea42dbd5-9452-44c2-8863-d46a21b0935a`，服务执行 119.76 秒。VHS 图像/音频参考、视频/音频 VAE、Qwen 编码器、量化 H3、原 LoRA 和 T8 采样、解码、BiRefNet、透明拼图全部完成；MP4 完整音视频解码通过，输出 22 帧，PNG 为 RGBA 3072×320，透明度含 0/255。采样监测的 GPU 峰值约 6250MiB，无 OOM；这不是 CUDA 内部分配的严格峰值，也不是画质验收。

随后恢复原图 0.4MP、6 秒的原生连线，实际输出尺寸为 864×480、158 帧、24fps、8 步，使用同一组试跑素材和种子：

| 配置 | 任务 ID | 实际结果 |
| --- | --- | --- |
| 原尺寸、不加 FFN 分块 | `a9dfaae8-cc54-4eb8-8e62-f0bf1d877a28` | 完成第 1 步后在 FFN 的 int8 `torch.cat` 申请 1.13GiB 时 GPU OOM，约 61.75 秒；主机仍有约 17GiB 可用内存。 |
| 原尺寸、FFN 两分块 | `9a51d9de-2c85-4e24-a291-fb9af9f7bef9` | 8 步采样、音视频解码、去背和拼图全部完成，服务执行 462.91 秒，无 OOM。 |

成功配置新增 `MiniMaxH3ChunkFeedForwardT8Advanced`，`chunks=2`、`seq_threshold=4096`，接在原 LoRA 与 T8 双时钟采样器之间，另接折叠的 `PreviewAny` 配置回执。它在打包 token 行数超过阈值时拆分 MLP 前向，不清空正在使用的模型、不修改磁盘核心；原 KJ 注意力补丁仍保留。分块会改变 GEMM 形状及浮点舍入，不声称逐位等价。回执还会改变模型加载的先后顺序，本次仅证明这组配置实跑成功，不将显存差值归因为单一节点或保证任何分辨率均能运行。

完整结果为带声 MP4（864×480、158 帧、约 6.583 秒）及 RGBA PNG（10368×2400）；完整音视频解码及 PNG 解码均通过。PNG 首帧角落 alpha 为 0，包含真实透明区域，而不只是添加通道。离散监控记录的 GPU 最高占用为 7298MiB，主机可用内存最低 16360MiB，服务 RSS 最高 9462MiB；不是严格 CUDA 峰值。输出位于 `output/video/H3_Local_Trial_20261021_original_size_ffn2_00001_.mp4` 和 `output/SpriteSheets/H3_Local_Trial_20261021_original_size_ffn2_00001_.png`，SHA256 及媒体规格见本地 `original-size-ffn2-result.json`。

完整运行通过后才把相同分块参数接回正式原图。最终为 103 节点、130 条边；相较逐字节备份，删除三个拆分节点，新增分块和回执两个节点，仅 251/254/256/257/370/443 六个保留节点因输入替换、兼容迁移或模型接线变化，其他原始节点、提示词、素材引用和生成参数保持不变。按输入名、端口和双向元数据检查全部连线通过。最终 SHA256 为 `d848ef43b42db4eddcf87e6149ca7405800a8466baea8577b54dd2f1d6eac732`。

本轮启动参数的回退副本为 `user/default/backups/h3-local-trial-20261021/docker-compose.override.before-trial.yml`，仅在队列为空且确认没有后续修改时恢复该文件并重建单个 comfyui 容器；无需删除模型或重建镜像。试跑脚本、私有素材及浏览器临时配置均未写入镜像；临时 Chrome 已关闭。

首个 1 步诊断任务虽生成视频，但旧 RMBG 控件导致拼图输出被服务拒绝；另一次陈旧前端导出也只执行 1 步。它们的失败/部分结果保留在本地记录，不冒充完整成功。小样与原尺寸基线使用真实原生 LiteGraph/graphToPrompt 导出；FFN 副本仅在该原生基线中加入原作两节点及对应连线，并在提交前断言步数和去背参数，未使用不完整的自制旁路转换器。

原图选择 FL2VA 权重配 ReferenceToVideo，代码允许执行，但不是[官方训练任务中 R2V 对应 Ref2VA 的配对](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native)。本次不擅自下载或替换另一套模型；运行通过不能推论参考跟随、动作或像素风格质量通过。全部试跑输入、原生导出、任务与资源证据保存在 `/opt/maoning/Reference/h3-local-trial-20261021/`，不提交人物素材、模型或临时浏览器数据。

## 主存换速度对照（2026-10-21）

固定上述原尺寸成功图、素材、seed20261021、模型精度、LoRA、8 步及 FFN2，仅比较原磁盘模式与当前主存策略。每次重启服务后完整执行，无节点缓存命中；未清系统页缓存或 swap，因此不是严格冷盘基准，也不是保留热模型的比较。

| 配置 | 完整耗时 | 后续步间中位数 | 任务 ID |
| --- | --- | --- | --- |
| 原磁盘模式 | 466.496 秒 | 48.810 秒 | `00456db5-483e-40b2-83db-5361e7928ccc` |
| 当前主存模式，首次 | 453.783 秒 | 45.151 秒 | `11a9f72c-54fc-423e-8cb3-44fc44608117` |
| 当前主存模式，复跑 | 454.794 秒 | 45.043 秒 | `c3f83cf9-31cc-465c-8b45-bd2a8bd8fbf6` |

完整耗时缩短约 2.5–2.7%，步间约 7.5–7.7%；不是成倍提速。约 6 秒同口径离散样本中，服务 RSS 从原模式最高 8039MiB 增至 22230–23212MiB。候选 2 秒保护样本最低可用内存 4800MiB，本次额外 swap 约 3.8–4.0GiB；两轮均无 OOM 或保护中断，不保证其他分辨率、参考数量及并发负载同样安全。测试看门狗不是常驻服务，不把原生压力阈值说成硬内存限制。

两轮均完成带声 864×480/158 帧视频、去背与 RGBA 10368×2400 拼图，完整解码和 FFN 回执通过。PNG 解码像素摘要在基线及两次候选之间均不同，原因尚未确定；未更换模型、精度、种子或采样设置，但不宣称逐位相同或完全无损。复跑后半段容器 GPU 遥测有缺口，不作为完整显存峰值；空队列时仅同配置重建本服务，最终 NVML、新 CUDA 小计算、3874 个节点及新版管理器 4.2.2 复验通过。

无预算主存驻留（允许 pin 且 cache-none）曾因主存余量降至约 2.5GiB、持续换出而被测试保护停止，未保留。原磁盘启动配置备份及本轮脚本、完整输出校验和资源记录在 `/opt/maoning/Reference/h3-ram-speed-20261021/`。回退时只在空队列下将 Compose 命令恢复为上述保守参数，保留 `--enable-manager`，再重建单个 comfyui；不覆盖后续用户修改，不删除模型或重建镜像。

## 完整 LoRA 与全量注意力（2026-10-21）

保留用户确认的当前 INT8 主模型、NVFP4 编码器、Turbo LoRA 和输入，不下载或换成另一套模型。用户当前原图已调整为 20 步，不以此前 8 步基准覆盖；当前参考是 `设定图高清.png`，不是旧试跑的 Hero 图和视频。

实际 0.33 的 `linear_input_act` 在 INT8+convrot 融合路径直接调用内核，可能绕过 `LoraLoaderBypassModelOnly` 注入的 fc2 forward，导致部分 LoRA 漏算；原生 MLP、KJ/T8 FFN 都走同一助手，不是两分块独有。`h3_fused_bypass.patch` 仅回移植[官方修复 d91ed5f / PR#16862](https://github.com/Comfy-Org/ComfyUI/pull/16862)：活动旁路的模块走完整 forward，普通模块仍保留融合路径，卸载 hook 时恢复原 flag。构建前校验旧文件 SHA，拒绝对未知版本静默套补丁；不复制宿主较新核心、不升级 PyTorch 或量化库。

候选镜像 `fc078d277f08cea43aa37a0fb48aef416c420d489c458efa0fda10e15d677be2`。离线构建复用原依赖；启动后 272 个包的版本、3874 个节点名称和 T8 的 373 个节点均与原服务相同。保留原 `h3-local` 镜像 `a9796897817f2ef121062bc9dba942e18135a163fe17a1833ee842f7b67297cf` 及原启动配置，可在空队列时仅恢复镜像标签并重建单个服务。

两镜像使用实际量化 MLP/LoRA 和 CUDA 各跑 64 项小张量对照，覆盖 FP16/BF16、有/无 bias、强度 0/0.8/1、INT8+convrot、1/2 分块，以及 CPU 权重卸载。旧镜像重现 fc2 漏算；候选与同块完整 forward 参考严格一致，最大误差 0，hook 生命周期检查通过。测试 allocated/reserved 峰值分别为 16.33/32MiB，不含驱动上下文，不是整图峰值。12 项路由测试与 6 项启动守卫也通过。当前 cu128 下 Comfy Kitchen 的优化 CUDA 后端仍禁用，不能把该测试说成 cu130 优化核验收。

工作流候选将 366 号 KJ Sage 节点换成原作 `MiniMaxH3LowVRAMAttentionT8Advanced(head_chunks=4)`，保留 FFN2/阈值4096。按头分组仍覆盖全部文本、图像参考、音频及视频 token，不使用稀疏注意力、抽帧、降分辨率或额外量化；不同内核及分组形状仍可能改变浮点舍入，不承诺逐像素相同。现 T8 版本仍有临时引用寿命较长的限制，不能把其“提前释放”描述为理想显存收益。

实际原生前端导出已逐节点比较：只变更注意力节点并新增回执，其他有效参数一致，保留 864×480、158帧、24fps、20步和当前 seed。旧 ImpactSwitch 保存后的端口顺序按名称迁移，保留选择值、旁路和全部旧分支，避免加载失败后误导出旧画布。正式原图现为103节点/131边，SHA256为`8fd5beb9d544099172f945bc1d1c69f1979251333aa8946f701d05b3cdfb1cd2`；写前核对原SHA无用户新编辑，写后静态验收和正式文件原生API再导出均通过，与实跑候选完全相同。

当前完整任务`2b755229-943b-4dce-851f-3244ba41c61d`成功执行1165.137秒（约19分25秒），20/20步、无缓存命中、节点错误、OOM或保护中断。注意力回执确认50/50块、foreign0、head4；FFN回执为2/4096，没有外部MLP委派。带声MP4为864×480、158帧、24fps，完整音视频解码通过；透明拼图为RGBA10368×2400、alpha0..255，完整解码通过，首中尾视频帧已查看。这证明当前规格可执行与计算链完整，不等同于噪点消失、BF16原精度或像素画美术质量验收。

580次保护采样中GPU最高7058MiB、服务RSS最高20825MiB、可用主存最低4601MiB，本次额外swap约3719MiB；不是CUDA严格峰值，也不保证更多参考、分辨率或并发负载可执行。成片位于`output/video/H3_Quality_20261021_head4_ffn2_00001_.mp4`，拼图位于`output/SpriteSheets/H3_Quality_20261021_head4_ffn2_00001_.png`。此前8步/旧素材的速度结果不能直接作为本次画质或性能A/B对照。

另以同一原生SDPA核对原生注意力、T8单组与四组，覆盖部分RoPE、FP16/BF16、1/17/129/257个token及旁路LoRA。本权重经文件头确认56头，32例全部严格一致。额外42头余数覆盖32例中有1例超出预设逐点容差（2/8256个元素；相对L2为0.00317），仍保留失败、未放宽阈值；全部头/token/QKV切片及hook计数通过，不能宣称任意头数逐位一致。首次夹具误带训练标志导致的RoPE拒绝已单独保留，只修随机夹具，不改生产代码。补充夹具守卫18项通过。实际Turbo文件头的259组A/B均匹配当前UNET，无目标或形状缺失；这是metadata核对，不冒充实时hook计数。

脚本、真实数值结果（包含未通过项）、原图逐字节备份、权限修复前记录和 Compose 回退副本保存在 `/opt/maoning/Reference/h3-quality-first-20261021/`。仅修复原 JSON 的组读写权限（600→660，其他用户仍无权限），不递归改权限、不修改模型。测试保护仅按任务 ID 中断自有任务，不中断用户队列。已打开的旧画布需重新打开该工作流文件；不自动覆盖用户浏览器中未保存的编辑。
