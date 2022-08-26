import 'package:dart_ml/src/tensor/core/tensor.dart';

Tensor<T> arange<T extends num>(int size) {
  var output = List<T>.generate(size, (index) => index as T);

  return Tensor<T>(output);
}
