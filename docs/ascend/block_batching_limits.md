# Block batching 当前能力与限制

实现版本 20，布局规划版本 2，runtime guard schema 3、语义契约版本 4。使用统一效果、控制流值域与布局计划。标准后端决定固定 factor 的资源可行性；只修改 TA。

## 访存依赖与保守回退（版本20）

- `tt.gather` 暂不支持，候选返回 `UNSUPPORTED_BATCH_OP`，选择完整 baseline。通用 `TritonOpConverter.cpp` 保持原实现，既有 gather lowering 问题不在本功能中修复。
- TTIR 改写不引入 HIVM 或 `PIPE_ALL`。guard 使用 `memory_policy="disjoint_or_terminal_store"`，已知地址访问对中只要有写，就继续要求实际字节区间不相交，同 root 与不同 root 均检查。
- 普通 device SSA 地址读取与写共存时，不再一律拒绝：若只有一个普通 store，所有 opaque load 与该 store 都在入口基本块中、load 在 store 之前，且输出值通过无 region、无内存效果的 SSA 计算依赖每条 opaque load 的结果，可沿既有数据依赖写回。该证明不通过其他 load、指针相同或源码顺序猜测依赖；存在 volatile 效果时不使用此放宽。
- native 计划用 `terminal_read_ids` 记录上述已证明的读取，绑定源/计划指纹及 binary 身份。guard 校验它完整覆盖 opaque 读取、唯一 store 及先后关系，再放行该读取的 device SSA 地址；不扫描 ids、不增加范围/唯一性或 allocation 容量查询。其余已知地址的读写、写写和布局条件仍逐调用检查。
- LK31 在 `MASK=None、LSE=None` 时，最后的 logits 间接读取经减去 lse 得到 logp，再写入唯一 LOG_P 输出，符合这条通用规则；embedding 的直接依赖写回也可使用。没有 kernel 名称或 shape 特判。
- 写后间接读、独立输出值、多个 store、region 内 opaque load（含 loop-carried index）及 volatile 混合仍返回 `UNPROVEN_MEMORY_NONINTERFERENCE`。LK31 的非空 LSE 会增加独立写回，暂未证明；非空 MASK 的 varying early return 仍受既有控制模型限制。
- `ShareInvariantLoad`、MemoryView/PointwiseRegion、动态 stride、uniform 控制、pointer roundtrip、访存属性和尾部保护继续使用原有模型。共享读取不会额外产生同步。已知地址原地/交错别名仍通过 guard 回退，完整 baseline 继承原编译器能力。
- Python/native 实现版本同步提升到20，语义契约版本为4，防止 v18 屏障计划或 v19 全拒绝计划被复用。完整 baseline 与 original-grid ABI 保留，候选执行失败不重跑 baseline。
- scatter、间接 volatile、循环 store、未知 pointer root、varying 控制流、一般 masked-load-without-other 及未建模 call/asm 仍不支持。
- v20 已在 A5-37 / torch-213 / 物理 NPU 5 完成重编与验证：820项 native、38项产品设备回归通过；原始 LK31 的 F=2/3、axis=1、图优化开/关共4个候选均实际 APPLIED，独立 CPU 参考最大绝对误差1.90735e-6。历史 v18/v19 性能数据仍属于各自构建，本轮未测性能。

以下版本条目为历史增量；与上述 v20 更新冲突的限制以 v20 为准。

版本14历史：通用SeededReduceConverter已移除；644个独立native、29个设备用例通过，原后端maxnum全NaN参考问题单列为1个strict XFAIL。性能结果另见本轮任务报告。版本13历史结果：本次属性增量已在 A5-37 / torch-271 验证：643项native、27项设备用例、5个Liger kernel各F=2/3共10个实际合并候选；设备使用taskqueue=1，下文既有双任务队列结果仍属于版本12。固定 factor 的资源可行性由标准后端编译判断；保留运行时 guard、完整 baseline 回退及 binary/ABI/缓存一致性。仅在 Triton-Ascend 实现，不依赖定制 NPUIR 补丁或资源报告。

- **普通读取的地址摘要可以缺省。** 已知 root 的未建模整数 SSA 可保留设备表达式；与写共存时需上述末尾依赖写回证明，否则回退。普通 store 不新增 scatter，pointer roundtrip 仍需已有证明。
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
- 支持一维线性前缀、静态 stride 的二维行视图，受支持的动态 stride 领先行轴视图，以及由单一原始 range 经有序 reshape 得到的同方向分解轴。生成 `index < min(N, origin + G*B)`，保留不足/过量 grid、两类尾部、非零逻辑起点、load other 和计算分组；不要求 N=G*B。原 scalar signed min/max/clamp 上界由同一 guard 精确求值，原地址中的 min/div 等未建模表达式仍不因此放开。
- 列方向仅传播通过验证的单基本块逐元素闭包，直接生成目标布局，不插入大转置再假设它会消失。统一 scalar 计算保持原域；共享/每 lane 小域值、reduce/scan、控制流或非兼容接口需要桥接时，本版保守 KEEP。二维/三维有限覆盖不等于任意 rank/encoding 支持；符号 stride 的列方向桥接仍返回 SYMBOLIC_LAYOUT_BRIDGE_UNSUPPORTED。
- load/store 独立证明后以有序 view 恢复语义形状。单行 RMSNorm 的已结构化 X/Y 与共享 W 继续 KEEP；多行 block RMSNorm 的动态 X/Y 行 stride 可在证明后使用 MemoryView，W 保持共享，各访问独立规划，无 kernel 名称特判。
- 只有实际采用新布局且标准 NPUIR 明确报告 UB 资源拒绝时，才从同一 frozen source、相同 F/axis 以 `semantic_only` 重编一次。该候选也失败则保留完整 baseline；内部 verifier、断言、崩溃和未知错误不借布局重试掩盖。首版不缓存失败尝试的负结果。
- metadata 的 `layout` 报告每访问的计划、来源、轴组、应用状态、桥接方式，以及整个 TA adapter 的 indirect/copy/transpose 观察计数。模块级计数不是逐访问归因，也不因合法内部 SIMT 统一拒绝候选。

## 参数化地址与动态 stride（版本15）

- 将程序/局部坐标与uniform标量参数分开：地址表示为参数化偏移加各坐标乘参数化系数。首批标量系数支持i32/i64参数、常量、加减与常数倍；有界规范化不调用通用求解器，不支持参数乘参数、未证明的截断或坐标乘坐标。
- 在原方向绑定之后证明 `coef(pid)=tileExtent*coef(local)`。例如 `(pid*M+r)*S+c` 对任意受guard接受的正S均能保持该关系；X/Y可采用不同的动态stride。独立参数之间需要额外等式条件的布局（例如P与M*S）仍KEEP，本版不生成这种关系的条件特化。
- 首批动态布局仅限已支持的领先行轴组、rank≤3、无循环的访存view；非合并轴的动态系数、复杂列方向桥接仍KEEP。共享值、计算域、reduce/scan分组、load other和访存属性不因布局改写扩大。
- 新pointer保留scalar基址，以动态stride缩放局部range；mask在一维完成比较后再广播，保留 `min(N, origin+G*B)`。load后保序恢复语义形状，store的值按相同映射折叠。
- 实际改写的符号布局在原v2 guard上追加版本1的 `layout_checks`。原guard继续检查源表达式位宽、内存效果和别名；附加guard检查正stride、新增宽整数/字节运算与指针范围，覆盖补齐的inactive lane。程序坐标使用合并后的grid，`grid`表达式仍读取原始hidden extents。不得用mask掩盖新增算术溢出。
- guard缓存包含原访问及新增布局检查的标量依赖；stride/边界/地址/grid变化重新验证。新增guard失败选择完整baseline。`semantic_only`不附加未实际应用布局的guard；编译期KEEP也不携带这些条件。
- 每访问报告增加 `symbolic_stride`、`element_stride` 与 `original_and_layout_runtime_guard` 证明范围。符号恒等式成功记为 PROVEN_SYMBOLIC_STRIDE_CHAIN。read-only分析报告不能授权执行；实际重写、标准后端编译和runtime guard仍分别确认。

## 任务队列配置

本版在 A5-37 / torch-213 验证了两种完整设备配置：

- `TRITON_ENABLE_TASKQUEUE=1`，PyTorch 使用默认任务队列配置。
- `TRITON_ENABLE_TASKQUEUE=0` 且 `TASK_QUEUE_ENABLE=0`，两端均采用同步提交配置；环境变量需在 Python 启动前设置。

仅关闭 Triton 队列、同时保留 PyTorch 默认异步队列时，观察到初始化值未被覆盖的间歇性输出异常；扩展前版本也有复现。直接 CANN launch 后的 stream synchronize 不能替代框架待提交任务的顺序保证。该混合配置本次未作为通过验收的组合，相关运行时互操作问题未在此次 i8/extern 增量中修复。需要同步模式时使用两端一致的配置；性能报告使用 taskqueue=1。

## 同类型归约 reshape 与连续标量写摘要（版本17）

- 仅在候选副本中消除 rank=1、完整输入/输出类型相同、无 encoding/efficient_layout、属性已验证且只有一个 axis=0 reduce 消费者的 allow_reorder reshape。选择 identity 排列，不将重排许可传播到新增 lane 维；一般 reshape、多消费者和布局提示继续保守拒绝。F=1 与拒绝后的原始模块不变。
- 对入口基本块中无 mask 的 scalar stores，识别同一 SSA pointer 的连续常量字段偏移0..K-1（2≤K≤64），生成一个逻辑[K]联合写效果。只改变guard的访问摘要，不合并实际store、不新增设备指令、不增加运行时guard种类。
- 联合效果保留在偏移0的原access ID；其余成员保留原slot、offset、pointer_checks，仅在效果摘要中设false mask。现有guard在应用mask前检查原算术；联合单射性继续拒绝短stride/重复grid写，跨根和额外访问仍检查真实alias。原layout规划与access ID、缓存标量依赖保持一致。
- 动态stride不假设等于字段数；孔洞、重复偏移、条件store、循环store或不同SSA基址不合组。循环控制流和数据相关gather的支持范围未扩大。
- Python/native实现版本同步为17，避免与既有v16独立循环实现归档混用；本版本基于本地v15范围，不包含v16循环支持。
