import 'dart:math';

import 'package:dart_ml/src/activation/activation_abstract.dart';
import 'package:dart_ml/src/tensor/functions.dart';
import 'package:dart_ml/src/tensor/tensor.dart';

class Sigmoid<T extends num> implements ActivationFunction<T> {
  @override
  Tensor<T> forward(Tensor<T> input) {
    var oneT = Tensor(1);
    return oneT / (oneT - exp(input * Tensor(-1)));
  }
}
