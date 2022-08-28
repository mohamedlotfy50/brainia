part of '../core/tensor.dart';

Tensor<T> ones<T extends num>(List<int> shape) {
  var size = _TensorHelper.initSize(shape);
  var output = List<T>.filled(size, 1 as T);

  return Tensor<T>(output)..reshape(shape);
}
