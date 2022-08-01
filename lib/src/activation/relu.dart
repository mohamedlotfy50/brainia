import 'package:dart_ml/src/activation/activation_abstract.dart';
import 'package:dart_ml/src/tensor/functions.dart';
import 'package:dart_ml/src/tensor/tensor.dart';

class ReLU<T extends num> implements ActivationFunction<T> {
  @override
  Tensor<T> forward(Tensor<T> input) {
    return maximum(0, input);
  }
}
