import 'package:dart_ml/src/tensor/core/tensor.dart';

abstract class NetworkLayer<T> {
  Tensor<T> forward(Tensor input);
}
