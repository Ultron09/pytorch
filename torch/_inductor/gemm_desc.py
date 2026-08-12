"""`GEMMDesc`: how to compute a GEMM, written down precisely enough to pin the bits.

A GEMM's floating-point result depends on the order its k products are added in,
because floating-point addition is not associative. Two kernels that compute the
same matrix product can therefore hand back different bits. `GEMMDesc` writes that
order down: give the same `GEMMDesc` to two machines and they owe you the same bits.

There are two ways to get one. A person writes it by hand, so a result stays
reproducible anywhere. Or a later layer produces one by reading what an existing
library would do for a shape, so a Triton kernel can be made to agree with that
library bit for bit. This module is only the description; the producers come later.

A `GEMMDesc` holds exactly two kinds of thing: which GEMM algorithm, and every
parameter that changes the floating-point order. It holds nothing that is only a
speed knob. Tile sizes, warp counts, pipeline depth, cluster shape and how much
scratch memory a library may use are all absent for that reason: they move the run
time, not the result.

Nothing here names a library, an algorithm id, or a vendor enum value. A library is
a *producer* of a `GEMMDesc`, never a part of one. Where an algorithm here happens
to equal a known library kernel, a comment says which one.

What is deliberately not a field
--------------------------------
Operand layout. Row-major, column-major and arbitrary strides say which bytes are
which matrix element; they describe the *problem*, not the algorithm. Reading an
element never rounds, so two runs with the same `GEMMDesc` over differently laid out
operands add the same products in the same order and return the same bits. Layout is
an input to a producer -- it is one of the things a library looks at when it picks a
kernel -- but it is not part of the description that comes out. Leaving it out is the
point: Inductor sees arbitrary layouts, and a descriptor that pinned one layout per
dtype would refuse work it can do.

The batch axis. A batched GEMM is a set of independent GEMMs, and nothing is ever
summed across the batch, so the batch cannot change the order of any one output
element. One `GEMMDesc` describes every GEMM in the batch. Only a GEMM that reduced
over the batch would need a field, and that is a different operation.

The values of alpha, beta and the bias. Those are data, and data does not belong in a
description of an algorithm. What changes the bits is *where* the single rounding to
`output_dtype` sits relative to them, and whether several constant factors are
multiplied together before they touch the accumulator or applied one at a time. Those
two questions are `epilogue` and `scale_apply`.

What this cannot say
--------------------
The k reduction here is a nest of cuts that ends in one chain of hardware
steps. Real kernels exist that do more than that -- for example a vector-wide load
whose elements form their own summation group underneath a lane's chain. A producer
that meets one must decline rather than write down the nearest `GEMMDesc`: a
nearly-right order is wrong bits, and losing coverage is much cheaper than answering
wrong.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import torch


class GEMMAlgorithm(Enum):
    """How operands become an update to the accumulator.

    These are the three ways real GEMM kernels do the multiply and the add. They
    round in three different places, so for the same inputs they are three
    different answers::

        MATRIX_INSTRUCTION         [sum of instruction_k products + acc] -> acc
                                   one rounding per instruction
        SCALAR_FUSED_MULTIPLY_ADD  fma(a, b, acc) -> acc
                                   one rounding per k element
        SCALAR_MULTIPLY_THEN_ADD   t = a * b, then acc + t -> acc
                                   two roundings per k element
    """

    # One hardware matrix-multiply instruction sums `instruction_k` products and
    # adds them to the accumulator; the whole step rounds once. This is what
    # Triton's `tl.dot` emits and what cuBLAS's tensor-core families do.
    MATRIX_INSTRUCTION = "matrix_instruction"

    # Scalar math, one k element per step: `acc = fma(a, b, acc)`, one rounding per
    # element. This is what cuBLAS's CUDA-core kernels do, and what Triton emits
    # with `enable_fp_fusion=True` when no matrix instruction is involved.
    SCALAR_FUSED_MULTIPLY_ADD = "scalar_fused_multiply_add"

    # The same scalar loop with the multiply and the add rounded separately:
    # `acc = acc + (a * b)`, two roundings per element. Triton reaches it with
    # `enable_fp_fusion=False`.
    SCALAR_MULTIPLY_THEN_ADD = "scalar_multiply_then_add"


class InputPrecision(Enum):
    """What the operands are rounded to before they are multiplied.

    fp32 operands can be fed to a matrix instruction in more than one way, and the
    choice changes the result while the dtype stays fp32. It cannot be read off
    `operand_dtype`, so it is its own field. The names are Triton's own
    `tl.dot(input_precision=...)` values.
    """

    # Operands are multiplied at their own precision. The only choice for anything
    # that is not fp32.
    IEEE = "ieee"

    # fp32 operands are rounded to tf32 (8-bit exponent, 10-bit mantissa) before the
    # multiply. Equal to cuBLAS's CUBLAS_COMPUTE_32F_FAST_TF32.
    TF32 = "tf32"

    # Each fp32 operand is split into three tf32 pieces and the pieces' products are
    # summed, which recovers most of the fp32 mantissa. Also known as 3xTF32.
    TF32X3 = "tf32x3"


class KCutLayout(Enum):
    """Which k elements one part of a k cut owns.

    With K = 12::

        CONTIGUOUS, span=4
          part0 = k[0:4]   part1 = k[4:8]   part2 = k[8:12]

        STRIDED, span=2, count=3
          tile:   t0     t1     t2     t3     t4     t5
                  k0 k1  k2 k3  k4 k5  k6 k7  k8 k9  k10 k11
          part0 = t0 t3    part1 = t1 t4    part2 = t2 t5

    Both hand three parts four k elements each, and the two answers differ, because
    which k elements share an accumulator is what decides the sum.
    """

    # Part i owns one run: k in [i * span, (i + 1) * span). The last part is short
    # when the length being cut is not a multiple of `span`.
    CONTIGUOUS = "contiguous"

    # The length being cut is divided into `span`-long tiles and part i owns every
    # tile j with j % count == i. cuBLAS's two-kernel gemv hands tiles out this way.
    STRIDED = "strided"


class MergeOrder(Enum):
    """The order the finished parts of a k cut are added together in."""

    # Part 0, then part 1, and so on: (((p0 + p1) + p2) + p3).
    SEQUENTIAL = "sequential"

    # A balanced binary tree: ((p0 + p1) + (p2 + p3)). A warp shuffle reduction is
    # this.
    PAIRWISE_TREE = "pairwise_tree"


class EpilogueOrder(Enum):
    """Where the single rounding to `output_dtype` sits relative to the epilogue.

    A bias added to the accumulator before that rounding and the same bias added
    after it give different bits, so this has to be said out loud, not assumed::

        NONE            [k sum] -round-> out
        IN_ACCUMULATOR  [k sum] -> *scale -> +bias -> +beta*C -round-> out
        AFTER_ROUNDING  [k sum] -round-> *scale -round-> +bias -round-> +beta*C
    """

    # Nothing happens after the k sum. The accumulator is rounded to `output_dtype`
    # once, on the store.
    NONE = "none"

    # Every epilogue step -- a scale multiply, a bias add, adding beta times C --
    # runs in `accumulate_dtype`, and the result is rounded to `output_dtype` once,
    # on the store.
    IN_ACCUMULATOR = "in_accumulator"

    # The accumulator is rounded to `output_dtype` first and the epilogue runs in
    # `output_dtype`, so every step rounds again.
    AFTER_ROUNDING = "after_rounding"


class ScaleApply(Enum):
    """How several constant scale factors reach the accumulator.

    Two factors multiplied together first cost one rounding; applied one at a time
    they cost two, and the results differ in the last bits whenever neither factor
    is a power of two.
    """

    # There are no constant scale factors.
    NONE = "none"

    # All factors are multiplied together first, then the accumulator is multiplied
    # once.
    FOLDED = "folded"

    # Each factor is its own multiply on the accumulator, so each one rounds.
    SEPARATE = "separate"


# An operand may be read at any of these. There is no fall-through: a dtype that is
# not listed is rejected, never quietly treated as one that is. That mistake is how a
# study of this problem in Triton ran an fp32 operand through the fp16 recipe and
# returned a wrong answer instead of declining.
FLOAT_OPERAND_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    }
)
INT_OPERAND_DTYPES: frozenset[torch.dtype] = frozenset({torch.int8, torch.uint8})
OPERAND_DTYPES: frozenset[torch.dtype] = FLOAT_OPERAND_DTYPES | INT_OPERAND_DTYPES

# What a k sum may be kept in. fp8 is missing on purpose: no hardware accumulates in
# it, and allowing it would let a producer write a descriptor nobody can honour.
ACCUMULATE_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float16, torch.float32, torch.float64, torch.int32, torch.int64}
)

OUTPUT_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int8,
        torch.int32,
    }
)

# What a partial sum may be stored in, and what a merge may add in.
MERGE_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    }
)


def _dtype_names(dtypes: frozenset[torch.dtype]) -> str:
    return ", ".join(sorted(str(d) for d in dtypes))


@dataclasses.dataclass(frozen=True)
class KCut:
    """One cut of the k axis, not one part of it.

    A cut says: take the k range handed to this level, slice it into parts, give each
    part its own accumulator starting at zero, and afterwards add the part totals
    together. The fields answer the five questions that has -- how long a part is
    (`span`), which k elements share a part (`layout`), how many parts (`count`), what
    order the part totals are added in (`merge`), and what dtypes the part total and
    the running merge sum are kept in.

    A cut is worth describing because an accumulator that restarts at zero puts a
    bracket in the sum. Take K = 12 with one matrix instruction per 2 k elements, so
    six instruction results d0..d5. With no cut, one accumulator walks all six::

        ((((( d0+d1 )+d2 )+d3 )+d4 )+d5 )

    With one cut at span 4, each group of two gets its own accumulator and the three
    totals are added::

        ( ( (d0+d1) + (d2+d3) ) + (d4+d5) )

    Same six numbers, every level still added left to right, different brackets.
    Floating-point addition is not associative, so those are different answers. This
    is not a corner case: the Triton study this type replaces first missed the second
    level because a flat sum matched the reference exactly while k fit in one group
    and parted from it at the first k that needed two.

    Cuts nest, outermost first. Level 0 cuts the whole k axis; level 1 cuts each part
    level 0 produced; and so on. With `k_cuts = (KCut(span=6, ...), KCut(span=2, ...))`
    over K = 12::

        K = 12
        |-- part k[0:6]              level 0, its own accumulator
        |   |-- part k[0:2]          level 1, its own accumulator
        |   |-- part k[2:4]
        |   +-- part k[4:6]
        |   total = ((k[0:2] + k[2:4]) + k[4:6])
        +-- part k[6:12]
            |-- part k[6:8]
            |-- part k[8:10]
            +-- part k[10:12]
            total = ((k[6:8] + k[8:10]) + k[10:12])
        result = k[0:6] total + k[6:12] total

    The innermost part has no cut under it. One accumulator walks it in index order,
    and `GEMMDesc.algorithm` and `GEMMDesc.instruction_k` say what one step of that
    walk is.
    """

    # k elements in one part for CONTIGUOUS, or in one tile for STRIDED. This is the
    # length itself and not a part count, because the length a library lands on is
    # rounded to a grain that no rule here could re-derive from the shape.
    span: int

    # The dtype a finished part is written at before it is merged, and the dtype the
    # running merge sum is kept in. They are two fields because all the useful
    # combinations are in use: fp32 partials summed in fp32, output-dtype partials
    # summed in fp32, and a chain kept in the output dtype the whole way. Assuming
    # the second one is a bug that hid for a long time in the Triton study this type
    # replaces -- the merge rounded to fp16 whatever the output dtype was, so every
    # bf16 split-k result was wrong.
    partial_dtype: torch.dtype
    merge_dtype: torch.dtype

    layout: KCutLayout = KCutLayout.CONTIGUOUS
    merge: MergeOrder = MergeOrder.SEQUENTIAL

    # How many parts. Only STRIDED needs it, because the tile-to-part rotation period
    # is the part count. For CONTIGUOUS the count follows from `span` and the length
    # being cut, so repeating it here could only ever disagree with it.
    count: int | None = None

    def __post_init__(self) -> None:
        if self.span < 1:
            raise ValueError(f"span must be at least 1, got {self.span}")
        contiguous = self.layout is KCutLayout.CONTIGUOUS
        if contiguous and self.count is not None:
            raise ValueError(
                "count must be None for a CONTIGUOUS cut: the number of parts "
                f"follows from span and the length being cut, got count={self.count}"
            )
        if not contiguous and (self.count is None or self.count < 2):
            raise ValueError(
                "a STRIDED cut needs count >= 2, since count is the tile "
                "rotation period and a cut into one part is not a cut, got "
                f"count={self.count}"
            )
        for name, dtype in (
            ("partial_dtype", self.partial_dtype),
            ("merge_dtype", self.merge_dtype),
        ):
            if dtype not in MERGE_DTYPES:
                raise ValueError(
                    f"unsupported {name} {dtype}; "
                    f"supported: {_dtype_names(MERGE_DTYPES)}"
                )


@dataclasses.dataclass(frozen=True)
class GEMMDesc:
    """Everything about a GEMM that decides its bits, and nothing else.

    The k sum is described from the outside in. `k_cuts` cuts the k axis into
    parts that each get their own accumulator, level by level, outermost first.
    Inside the last part one accumulator runs a chain of steps in index order, and
    `algorithm`, `instruction_k` and `k_loop_step` say what a step is and where the
    short one sits. `epilogue` and `scale_apply` say what happens after the sum.
    Everything else here is a dtype.

    A `GEMMDesc` is a value: it is frozen, it compares and hashes by its fields, and
    there is no module state anywhere near it. Two descriptors that compare equal
    must produce the same bits, so it can be used as a cache key without a second
    thought.
    """

    # The dtype both operands are read at. One field and not two, because a GEMM with
    # two different operand dtypes is out of scope; when that changes this splits in
    # two rather than growing a rule about which one wins.
    operand_dtype: torch.dtype

    # The dtype the k sum is kept in. An explicit field because it is not implied by
    # the operand dtype: fp16 operands are accumulated in fp32 by almost everything
    # but not by everything, and fp32 and fp64 operands make "always fp32" wrong.
    accumulate_dtype: torch.dtype

    # The dtype the finished value is rounded to on the store.
    output_dtype: torch.dtype

    algorithm: GEMMAlgorithm

    input_precision: InputPrecision = InputPrecision.IEEE

    # How many k elements one matrix instruction sums into the accumulator in a
    # single rounding. It is the step size of the innermost chain, so it sets where
    # that chain rounds, which is why it is here rather than left to the compiler.
    #
    # It is not the loop's k step, and seeing why is also why the loop's k step is
    # not a field at all. With a threadblock k step of 64 and instruction_k 16, one
    # turn of the loop issues four instructions into the same accumulator:
    #
    #     turn 0:  [16][16][16][16]  -> added into acc one after another
    #     turn 1:  [16][16][16][16]  -> added into acc one after another
    #
    # The chain steps by 16 either way. Widening the turn to 128 regroups the loop
    # but not the chain, so it cannot move the bits.
    instruction_k: int | None = None

    # Where a matrix instruction's result meets the running accumulator:
    #
    #     True   a, b -> [instruction sums and adds into acc] -round-> acc
    #
    #     False  a, b -> [instruction sums into zero]         -round-> part
    #            acc + part                                   -round-> acc
    #
    # True is `tl.dot(a, b, acc)` and rounds once a step; False is
    # `acc + tl.dot(a, b)` and rounds twice. The two scalar algorithms draw the same
    # distinction in their own names, so this is their matrix-instruction twin: it is
    # required for MATRIX_INSTRUCTION and None otherwise. The name is torch's -- the
    # `use_fast_accum` argument of `torch._scaled_mm`, and `USE_FAST_ACCUM` in
    # Inductor's GEMM templates.
    #
    # That these are two different sums is measured, not assumed. On a GB300 they
    # always agree, but only because Triton rewrites the second into the first before
    # codegen -- `CombineDotAddFPattern` in `lib/Dialect/Triton/Transforms/Combine.cpp`
    # turns `addf(dot(a, b, 0), acc)` into `dot(a, b, acc)` -- so both reach the same
    # machine code. Block that rewrite and they part company in every dtype: over a
    # 640x512 output at K = 4096, 326,703 of 327,680 elements differ for fp16 operands
    # with an fp32 accumulator. The rewrite is switched off by design on sm_90 when
    # both operands are fp8, which is exactly where torch hands the choice to the
    # caller. There is no default here, for the same reason the dtypes have none: it
    # decides bits, so it has to be stated.
    use_fast_accum: bool | None = None

    # The k loop step of one accumulator: how many k elements one turn of the
    # mainloop covers. It is here for one reason. When a part is not a whole number
    # of turns, a kernel that runs the short turn FIRST puts the instruction group
    # boundaries somewhere else than one that runs it last. With K = 10 and
    # instruction_k 4:
    #
    #     None, short group last    [k0 k1 k2 k3][k4 k5 k6 k7][k8 k9]
    #                                     d0           d1        d2
    #                               boundaries at k = 0, 4, 8
    #
    #     4, short group first      [k0 k1][k2 k3 k4 k5][k6 k7 k8 k9]
    #                                  d0        d1          d2
    #                               boundaries at k = 0, 2, 6
    #
    # Same K, same instruction_k, three groups either way -- but different k
    # elements are folded into the same rounding, so the two are different sums.
    # None means the boundaries run on an even grid from 0 and the short group is
    # last.
    k_loop_step: int | None = None

    # The nest of k cuts, outermost first. Empty means one accumulator walks
    # the whole k axis, which is the ordinary non-split GEMM.
    k_cuts: tuple[KCut, ...] = ()

    epilogue: EpilogueOrder = EpilogueOrder.NONE

    scale_apply: ScaleApply = ScaleApply.NONE

    def __post_init__(self) -> None:
        self._check_dtypes()
        self._check_algorithm()
        self._check_cuts()
        self._check_epilogue()

    @property
    def is_floating_point(self) -> bool:
        """Whether this GEMM rounds at all. An integer GEMM does not.

        Integer addition is exact and associative, so for an integer GEMM the
        ordering fields describe the kernel but do not constrain the answer.
        """
        return self.operand_dtype.is_floating_point

    def _check_dtypes(self) -> None:
        for name, dtype, allowed in (
            ("operand_dtype", self.operand_dtype, OPERAND_DTYPES),
            ("accumulate_dtype", self.accumulate_dtype, ACCUMULATE_DTYPES),
            ("output_dtype", self.output_dtype, OUTPUT_DTYPES),
        ):
            if dtype not in allowed:
                raise ValueError(
                    f"unsupported {name} {dtype}; supported: {_dtype_names(allowed)}."
                    " An unlisted dtype is rejected rather than run under some other"
                    " dtype's recipe, which would return a wrong answer quietly."
                )
        kinds = {
            self.operand_dtype.is_floating_point,
            self.accumulate_dtype.is_floating_point,
            self.output_dtype.is_floating_point,
        }
        if len(kinds) != 1:
            raise ValueError(
                f"operand_dtype {self.operand_dtype}, accumulate_dtype "
                f"{self.accumulate_dtype} and output_dtype {self.output_dtype} must "
                "all be floating point or all be integer"
            )

    def _check_algorithm(self) -> None:
        matrix = self.algorithm is GEMMAlgorithm.MATRIX_INSTRUCTION
        if matrix and (self.instruction_k is None or self.instruction_k < 1):
            raise ValueError(
                "MATRIX_INSTRUCTION needs instruction_k >= 1, the number of k "
                "elements one instruction sums in a single rounding, got "
                f"{self.instruction_k}"
            )
        if not matrix and self.instruction_k is not None:
            raise ValueError(
                f"instruction_k must be None for {self.algorithm.name}: a scalar step"
                f" always covers exactly one k element, got {self.instruction_k}"
            )
        if not matrix and self.k_loop_step is not None:
            raise ValueError(
                f"k_loop_step must be None for {self.algorithm.name}: with one "
                "rounding per k element there are no instruction groups for a short "
                "first turn to shift"
            )
        if matrix and self.use_fast_accum is None:
            raise ValueError(
                "MATRIX_INSTRUCTION needs use_fast_accum: adding into the accumulator "
                "and adding into zero then merging are two different sums, and there "
                "is no safe default to pick for you"
            )
        if not matrix and self.use_fast_accum is not None:
            raise ValueError(
                f"use_fast_accum must be None for {self.algorithm.name}: a scalar "
                "algorithm already says whether the multiply and the add round "
                f"together, got {self.use_fast_accum}"
            )
        if self.k_loop_step is not None and self.k_loop_step < 1:
            raise ValueError(f"k_loop_step must be at least 1, got {self.k_loop_step}")
        self._check_input_precision(matrix)

    def _check_input_precision(self, matrix: bool) -> None:
        if self.input_precision is InputPrecision.IEEE:
            return
        if self.operand_dtype is not torch.float32:
            raise ValueError(
                f"input_precision {self.input_precision.name} needs float32 operands,"
                f" got {self.operand_dtype}: it names how fp32 is cut down before the"
                " multiply"
            )
        if not matrix:
            raise ValueError(
                f"input_precision {self.input_precision.name} needs "
                f"MATRIX_INSTRUCTION, got {self.algorithm.name}: tf32 exists only as "
                "a matrix instruction input"
            )

    def _check_cuts(self) -> None:
        if not isinstance(self.k_cuts, tuple):
            raise ValueError(
                "k_cuts must be a tuple so the descriptor stays hashable, got "
                f"{type(self.k_cuts).__name__}"
            )
        for level, part in enumerate(self.k_cuts):
            if not isinstance(part, KCut):
                raise ValueError(
                    f"k_cuts[{level}] must be a KCut, got "
                    f"{type(part).__name__}"
                )
            self._check_cut_dtypes(level, part)

    def _check_cut_dtypes(self, level: int, part: KCut) -> None:
        for name, dtype in (
            ("partial_dtype", part.partial_dtype),
            ("merge_dtype", part.merge_dtype),
        ):
            if dtype.is_floating_point != self.is_floating_point:
                raise ValueError(
                    f"k_cuts[{level}].{name} is {dtype}, which does not match "
                    f"an operand_dtype of {self.operand_dtype}"
                )

    def _check_epilogue(self) -> None:
        scaled = self.scale_apply is not ScaleApply.NONE
        if scaled and self.epilogue is EpilogueOrder.NONE:
            raise ValueError(
                f"scale_apply {self.scale_apply.name} needs an epilogue: a scale "
                "multiply is an epilogue step, so epilogue must say whether it runs "
                "before or after the rounding to output_dtype"
            )
