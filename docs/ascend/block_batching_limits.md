# Block batching 当前能力与限制

实现版本 14：通用SeededReduceConverter已移除；644个独立native、29个设备用例通过，原后端maxnum全NaN参考问题单列为1个strict XFAIL。性能结果另见本轮任务报告。版本13历史结果：本次属性增量已在 A5-37 / torch-271 验证：643项native、27项设备用例、5个Liger kernel各F=2/3共10个实际合并候选；设备使用taskqueue=1，下文既有双任务队列结果仍属于版本12。固定 factor 的资源可行性由标准后端编译判断；保留运行时 guard、完整 baseline 回退及 binary/ABI/缓存一致性。仅在 Triton-Ascend 实现，不依赖定制 NPUIR 补丁或资源报告。

- **地址表达式中的除法暂不支持。** `arith.divsi` 等整数除法在逐元素计算中已有 Basic 模型，但不能因此用于地址证明。当前地址摘要/guard 只接收已建模的仿射运算及转换；例如 `row = offset / n_cols` 用于 load 地址时，返回 `UNSUPPORTED_ADDRESS_EXPRESSION` 并使用完整 baseline。该限制本版保留。
- **新增 i8 为受限字节数据闭包。** 允许常量、普通 load/store、cmp/select、已支持的 shape 操作及 byte-pointer 地址计算；不开放一般 i8 算术、转换、归约、scalar i8 host ABI 或 inttoptr。标准 bool 读取仅接受 `ptr<i1>→ptr<i8>` 后直接带 `was_bool_to_int8=true` 的 load，按该规范化链的一个存储字节解释地址，保留属性。未标记的任意 i1 访存布局仍拒绝。
- **外部逐元素函数仍属于 E 类，只有已知子集开放。** 当前只接受空 libname/libpath、pure=true 的 `__hmf_pow_fp32(f32,f32)->f32` 和 `__hmf_tanhf(f32)->f32`。保留原 symbol、精度与逐元素语义；新无效 lane 的输入填 1，以保证 pow/tanh 在安全域求值。其他 symbol、库路径、签名、impure 调用及调用型归约 combiner 均不开放。
- 有效 factor 上限为 64；候选可能因 UB 等标准后端资源限制失败。支持某 op/dtype 不保证所有组合可编译，也不保证比 baseline 更快。
- 每个 B op 有受限正例，不等于所有 dtype、rank、属性和访存用途均可组合。


## 归约范围（版本14）

- 通用TritonToLinalg不再注册或构建SeededReduceConverter；F=1及合并后的普通maxnum/minnum恢复既有lowering，不新增顺序fold。
- block batching不再接受带cmpf uno的NaN-aware reduce combiner，返回UNSUPPORTED_NAN_REDUCTION_COMBINER；tuple reduce直接返回外部捕获值时返回UNSUPPORTED_REDUCTION_RETURN_CAPTURE。已编译的完整baseline可用时仍按既有规则回退。
- 实测原后端maxnum全NaN行返回-inf而非NaN，合并候选与baseline一致；该既有参考失败明确保留，不计入通过数。有限/混合NaN行单独验证。
- 原TA对特定NaN/tie输入的行为不在本功能中修复；原baseline本身失败的组合不能承诺回退可执行。普通数值的max/min和sum等归约继续验证；特殊值能力以实际标准后端结果为准。

## 访存属性与轴提示（版本13增量）

- 普通 load/store 接受已有 Triton cache/eviction 枚举，并在语义改写和布局快路径保留原属性；后端仍可按原有能力拒绝配置。
- volatile load 保留 isVolatile，结果按 lane 变化处理，不能将共享地址的一次读取广播给所有原 program。guard 额外要求每条 volatile 访问在原 grid、局部坐标和循环域上不重复地址；无法证明时返回 OVERLAPPING_VOLATILE_READS 并完整回退。该限制保守地排除共享 volatile、广播地址及重复循环读取，不扩大为跨 program 通信支持。
- 支持直接 SSA make_block_ptr / advance 链的 boundary_check。候选内部使用现有 tensor-pointer normalizer 将其转换为显式上下界 mask 和普通指针，保留 zero/NaN padding；未指定 padding 的屏蔽值选择零。循环携带、select、call 等尚未证明的 block pointer 流在进入 normalizer 前拒绝；普通指针上的异常 boundary/padding 不直接放行。
- tt.contiguity、tt.divisibility、tt.constancy 按原结果 rank 检查一维正 i32 向量格式后，从候选中保守移除，避免维度提升后沿用失效提示。后续分析可重新推导；可能影响性能。未知属性和非默认 overflow/fastmath 承诺继续拒绝。
- 实现版本同时用于 Python/native 一致性和候选缓存身份；旧 binary、ABI 和完整 baseline 规则保持。

## 按需合轴布局

F>1 默认使用 `block_batching_layout="auto"`；`semantic_only` 保留统一前置 F 的语义表示，用于同构建对照和一次编译期降级。F=1 归一为关闭，不进入布局规划。策略和实现版本参与缓存身份，logical lane_width 仍为 F，原 grid、pid/num_programs 与完整 baseline 不变。

- 从带边界的原始逻辑坐标绑定方向，再证明每访问的字节 stride 链。不会仅因两个轴 stride 相等而跨方向合并。方向不明、孔洞 mask、未知转换或超过分析预算时 KEEP。
- 支持一维线性前缀、静态 stride 的二维行视图，以及由单一原始 range 经有序 reshape 得到的同方向分解轴。生成 `index < min(N, origin + G*B)`，保留不足/过量 grid、两类尾部、非零逻辑起点、load other 和计算分组；不要求 N=G*B。原 scalar signed min/max/clamp 上界由同一 guard 精确求值，原地址中的 min/div 等未建模表达式仍不因此放开。
- 列方向仅传播通过验证的单基本块逐元素闭包，直接生成目标布局，不插入大转置再假设它会消失。统一 scalar 计算保持原域；共享/每 lane 小域值、reduce/scan、控制流或非兼容接口需要桥接时，本版保守 KEEP。二维/三维有限覆盖不等于任意 rank/encoding 支持；动态地址 stride 的符号消去尚未覆盖。
- load/store 独立证明后以有序 view 恢复语义形状。RMSNorm 的 X/Y、共享 W、独立行标量继续分别 KEEP，无 kernel 名称特判。
- 只有实际采用新布局且标准 NPUIR 明确报告 UB 资源拒绝时，才从同一 frozen source、相同 F/axis 以 `semantic_only` 重编一次。该候选也失败则保留完整 baseline；内部 verifier、断言、崩溃和未知错误不借布局重试掩盖。首版不缓存失败尝试的负结果。
- metadata 的 `layout` 报告每访问的计划、来源、轴组、应用状态、桥接方式，以及整个 TA adapter 的 indirect/copy/transpose 观察计数。模块级计数不是逐访问归因，也不因合法内部 SIMT 统一拒绝候选。

## 任务队列配置

本版在 A5-37 / torch-213 验证了两种完整设备配置：

- `TRITON_ENABLE_TASKQUEUE=1`，PyTorch 使用默认任务队列配置。
- `TRITON_ENABLE_TASKQUEUE=0` 且 `TASK_QUEUE_ENABLE=0`，两端均采用同步提交配置；环境变量需在 Python 启动前设置。

仅关闭 Triton 队列、同时保留 PyTorch 默认异步队列时，观察到初始化值未被覆盖的间歇性输出异常；扩展前版本也有复现。直接 CANN launch 后的 stream synchronize 不能替代框架待提交任务的顺序保证。该混合配置本次未作为通过验收的组合，相关运行时互操作问题未在此次 i8/extern 增量中修复。需要同步模式时使用两端一致的配置；性能报告使用 taskqueue=1。
