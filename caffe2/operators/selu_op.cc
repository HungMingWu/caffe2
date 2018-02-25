#include "caffe2/operators/selu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool SeluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.size());
  EigenVectorArrayMap<float> Yvec(Y->mutable_data<float>(), Y->size());
  Yvec = lambda_ * (Xvec > 0).select(Xvec, (alpha_ * Xvec.exp() - alpha_));
  return true;
}

REGISTER_CPU_OPERATOR(Selu, SeluOp<float, CPUContext>);

// Input: X; output: Y
OPERATOR_SCHEMA(Selu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function, y = scale*(alpha_*e^x-alpha_ if x < 0 else x),
is applied to the tensor elementwise.
)DOC")
    .Arg(
        "alpha",
        "(float) default to 1.6732~; affects the activation function itself. "
        "This should go with the weight initialization in the paper. "
        " See https://arxiv.org/abs/1706.02515 ")
    .Arg(
        "scale",
        "(float) default to 1.0507~; affects the activation function itself.")
    .Input(0, "X", "input tensor")
    .Output(0, "Y", "input tensor");


} // namespace caffe2
