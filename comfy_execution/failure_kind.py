"""机器可读的失败分类。

为什么单独一个模块：`execution.py` 与 `comfy_execution/graph.py` 都要产出
`error_details`，而 graph.py 要避免在早期就拉进 torch。所以这里只放**零依赖的
判定与文案**，把「是不是 OOM」这个需要 model_management 的判断交给调用方传进来。

分工（改动前先说清楚，否则这一层会被误用）：

* 服务端只报**事实** —— 失败属于哪一类（failure_kind）、是否属于「资源条件变化后
  存在成功可能」的一类（retryable）、以及关于这一类的说明（hints）。
* 服务端**不给重试建议**。它不知道客户端的尺寸、批次数、显存余量，也不知道失败
  落在哪个节点上对客户端意味着什么；替客户端做决定等于把错误决策挪个位置。
* retryable=True 的含义严格限定为上面那句「这一类失败在资源条件变化后存在成功
  可能」，它**不**表示「现在重试就会成功」，更不表示「建议重试」。典型的反例：
  采样期 OOM 也属于 out_of_memory（retryable=True），但只把 VAE 解码分块对采样期
  峰值毫无作用 —— 客户端必须结合 node_type 自己判断。
"""

from typing import Iterable, Optional

FAILURE_KIND_OUT_OF_MEMORY = "out_of_memory"
FAILURE_KIND_SHAPE_MISMATCH = "shape_mismatch"
FAILURE_KIND_INTERRUPTED = "interrupted"
FAILURE_KIND_EXECUTION_BLOCKED = "execution_blocked"
FAILURE_KIND_DEPENDENCY_CYCLE = "dependency_cycle"
FAILURE_KIND_INTERNAL = "internal"

ALL_FAILURE_KINDS = (
    FAILURE_KIND_OUT_OF_MEMORY,
    FAILURE_KIND_SHAPE_MISMATCH,
    FAILURE_KIND_INTERRUPTED,
    FAILURE_KIND_EXECUTION_BLOCKED,
    FAILURE_KIND_DEPENDENCY_CYCLE,
    FAILURE_KIND_INTERNAL,
)

#: 「资源条件变化后存在成功可能」的失败类型。见模块开头的限定说明。
RETRYABLE_KINDS = frozenset({FAILURE_KIND_OUT_OF_MEMORY})

_HINTS = {
    FAILURE_KIND_OUT_OF_MEMORY: (
        "显存不足（OOM）。服务端已卸载全部已加载模型，下一个任务会重新加载模型。",
        "这类失败与工作流结构无关，只与失败时刻的显存占用相关（输出尺寸、每批张数、其它进程都算）。",
    ),
    FAILURE_KIND_SHAPE_MISMATCH: (
        "张量形状不匹配。通常出现在采样器接到的 CLIP / 模型与工作流不配套时（Load CLIP 一类节点的文件或 type 选错）。",
    ),
    FAILURE_KIND_INTERRUPTED: (
        "该任务被中断（cancel / interrupt），不是执行错误。",
    ),
    FAILURE_KIND_EXECUTION_BLOCKED: (
        "该节点被 ExecutionBlocker 主动阻断，不是异常。阻断原因见 exception_message。",
    ),
    FAILURE_KIND_DEPENDENCY_CYCLE: (
        "工作流存在依赖环：节点之间互相依赖，无法确定执行顺序。",
    ),
    FAILURE_KIND_INTERNAL: (
        "未归类的内部错误。具体异常类型见 exception_type，调用栈见 traceback。",
    ),
}


def failure_metadata(failure_kind: str) -> dict:
    """把失败类型展开成要随 `error_details` 一起发的三个字段。"""
    return {
        "failure_kind": failure_kind,
        "retryable": failure_kind in RETRYABLE_KINDS,
        "hints": list(_HINTS.get(failure_kind, ())),
    }


def hint_lines(failure_kind: str) -> Iterable[str]:
    return _HINTS.get(failure_kind, ())


def is_shape_mismatch(ex: BaseException, class_type: Optional[str] = None) -> bool:
    """mat1/mat2 形状不匹配 —— 改动前这是 `execution.py` 里的一条 elif 就地判断。"""
    if not isinstance(ex, RuntimeError):
        return False
    if "mat1 and mat2 shapes" not in str(ex):
        return False
    return class_type is None or "Sampler" in class_type


def classify_exception(
    ex: BaseException,
    *,
    is_oom: bool,
    class_type: Optional[str] = None,
    interrupted: bool = False,
) -> str:
    """把异常归到一个 failure_kind。

    `is_oom` 由调用方用 `comfy.model_management.is_oom()` 算好传进来 —— 这里不
    import torch，也不重复实现一遍那套 isinstance 判断（重复实现必然漂移）。
    """
    if interrupted:
        return FAILURE_KIND_INTERRUPTED
    if is_oom:
        return FAILURE_KIND_OUT_OF_MEMORY
    if is_shape_mismatch(ex, class_type):
        return FAILURE_KIND_SHAPE_MISMATCH
    return FAILURE_KIND_INTERNAL
