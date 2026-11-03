// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_SCHEDULE_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_SCHEDULE_H
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include "TritonToGraph/BlockBatching/BatchRangeAnalysis.h"
#include <cstdint>
#include <optional>
#include <string>
namespace mlir::triton::cfg {
// Layer 2 contract: every entry point in this header is a pure predicate.
// It takes facts and returns a verdict. It must not take a ModuleOp, must not
// create or erase any operation, and must be callable before any rewrite has
// happened. Mirrors llvm::isSafeToUnrollAndJam.

// How one program's rows sit on the logical row axis. Only these two are
// recognised; anything else is rejected rather than guessed.
//
//   Cyclic   成块发：lower = p*R, step = G*R。一圈取连续 R 行，圈与圈相隔
//            G*R 行，所以圈之间**不**连续。合并要把 R 放大到 R*F，起点与步长
//            同步放大，grid 不变。
//   Blocked  定长分块：步长是正常数 c。每个原程序守着自己的一段连续行，
//            lower = p*R、upper = min((p+1)*R, N)。折叠后 lane u 整个接管原程序
//            g*F+u 那一段，步长不动——圈数各份可能差一个尾部，靠掩码处理。
//
// 判别依据是步长而不是起点：成块发的步长含 num_programs，定长分块的是常数。
//
// 2026-11-02 原文留档。Blocked 这一条原本写的是：「步长是正常数 c。圈与圈本来
// 就相邻，合并只把步长变成 c*F，边界一个字不改。坐标归一对它是空操作，这里只
// 负责认出来。」那描述的是不折叠的手写参照版（grid 不变、一个程序的行段按 F 行
// 一组切开），与本设计服从的 grid 契约 v3（grid[axis] /= F）不符。按它写出来的
// 改写会重新分组浮点累加，判定层因此长出了关卡 1.5a/1.5b 两道精度闸；订正后
// 两关一并撤销，换成下面这条「段首段尾必须能按 lane 算出来」的要求。
struct BatchLogicalAxis {
  enum Distribution : uint8_t { Cyclic, Blocked };
  Distribution distribution = Cyclic;
  unsigned loopID = 0;
  int gridAxis = -1;
  // 一圈覆盖几行。成块发是 R，定长分块恒为 1（它靠步长走，不靠块宽）。
  int64_t chunk = 1;
};

struct BatchLoopScheduleResult {
  std::optional<BatchLogicalAxis> axis;
  std::string reason; // 空表示成功
};

// 入参是一条循环而不是一组访存：归一是循环的属性。"函数里有且只有一条程序轴
// 循环、且在顶层"由 BatchLoopLegality 的形态检查负责，不在这里。
//
// 不收 BatchAccessFact，因为它带 Operation*，头文件会把整个 MLIR 方言和
// llvm/Support/JSON.h 拖进判定层，破掉 C2/C3，也就没法脱离 MLIR 单测了。
// 不收 factor：用到 F 的只有圈数门槛 T >= F 与 F <= R，那是主谓词的事。
BatchLoopScheduleResult
normalizeBatchLoopCoordinates(const BatchLoopFact &loop,
                              const BatchRangeFacts &ranges, int gridAxis);

// 把访存表达式里这条循环的循环变量换成「下界 + 圈内偏移」，依据是恒等式
//
//     Induction(id)  ≡  loop.lower  +  Induction(id)'
//
// 右边的 Induction 代表圈内偏移（0, step, 2*step, …）。按 lane 不同的部分全部
// 集中在 lower 里（成块发与定长分块的 lower 都是 p*R，step 各份相同），所以代换
// 之后程序号显式出现在地址里，圈内偏移则是一个各份相同的量。
//
// 存在的理由：布局分析整套机制挂在"byteOffset 里有没有 K::Program"上，而带程序轴
// 循环时程序号住在循环下界里，不在地址里——于是一条按行走的访存会被判成
// SHARED_ACCESS（"不随程序轴变化"）。代换是把它接上去的那一步。
//
// 返回空表示代换不下去（lower 缺失），调用方应按证不出处理，不要退而用原式。
// 不含这条循环变量的表达式原样返回，指针不变。
BatchExprRef normalizeBatchAccessOnLogicalAxis(const BatchExprRef &e,
                                               const BatchLoopFact &loop);
} // namespace mlir::triton::cfg
#endif
