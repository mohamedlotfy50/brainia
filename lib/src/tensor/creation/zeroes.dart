part of '../core/tensor.dart';

Tensor<T> zeros<T extends num>(List<int> shape) {
  var size = _TensorHelper.initSize(shape);
  var output = List<T>.filled(size, 0 as T);

  return Tensor(output)..reshape(shape);
}
