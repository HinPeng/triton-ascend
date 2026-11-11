// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopLegality.h"

namespace mlir::triton::cfg {

BatchLoopLegality isSafeToBatchLoop(const BatchLoopFact *programLoop,
                                    const BatchRangeFacts &ranges,
                                    int gridAxis) {
  // 关卡 0 的结论由调用方给出（BatchLoopForm.h）。没有程序轴循环是合法形态而
  // 不是失败：直线 kernel 与只含块内循环的 kernel 都走这一支，交回既有的程序级
  // 加宽路径，循环一条都不动。
  if (!programLoop)
    return {true, "", std::nullopt};

  // 关卡 1：坐标归一。
  auto schedule = normalizeBatchLoopCoordinates(*programLoop, ranges, gridAxis);
  if (!schedule.axis)
    return {false, schedule.reason, std::nullopt};

  // 两种分布都不跨行：lane u 恒等于原程序 g*F+u，处理的行、顺序、写回的槽位全都
  // 原样不动，所以带多少个归约、带多少条跨行内存依赖都一样逐位一致。
  //
  // 2026-11-02 之前这里还有两关，只针对定长分块：1.5a 见到 carriesValues 就报
  // LOOP_REDUCTION_CROSSES_ROWS，1.5b 对其余的报 LOOP_DISTANCE_UNPROVEN。两关的
  // 共同前提是"定长分块把一个程序的行段按 F 行一组切开"，那是不折叠手写版的做法
  // （步长乘 F、grid 不变）。本设计服从 grid /= F，分块是按**原程序**切的，一条
  // 行段整个归一条 lane。数值实验：
  // block_batching_audit/20261102_blocked_folding/check_blocked_bitexact.py，
  // 折叠版 6/6 形状逐位一致，不折叠的 step*F 版 6/6 都不一致。
  //
  // 替代它们的是关卡 1 里定长分块新增的边界形态要求：段首段尾必须能按 lane 算出
  // 来，否则报 UNPROVEN_LANE_SEGMENT_BOUND。
  return {true, "", schedule.axis};
}

} // namespace mlir::triton::cfg
