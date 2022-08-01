import 'package:dart_ml/src/tensor/tensor.dart';

abstract class ActivationFunction<T extends num> {
  Tensor forward(Tensor<T> input);
}
