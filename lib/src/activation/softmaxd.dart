import 'package:dart_ml/src/activation/activation_abstract.dart';
import 'package:dart_ml/src/tensor/functions.dart';
import 'package:dart_ml/src/tensor/tensor.dart';

class Softmax<T extends num> implements ActivationFunction<T> {
  @override
  Tensor<T> forward(Tensor<T> input) {
    var e = exp(input - max(input, axis: 1, keepDims: true));
    return e / sum(e, axis: 1, keepDims: true);
  }
}
