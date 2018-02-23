#include "caffe2/operators/negate_gradient_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(NegateGradient, NegateGradientOp<CPUContext>);
OPERATOR_SCHEMA(NegateGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
NegagteGradient operator in forward pass simply copies input to the
output, and in backward pass, flips the sign of the output gradient
)DOC");

} // namespace caffe2
