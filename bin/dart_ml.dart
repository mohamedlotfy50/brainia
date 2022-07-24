import 'dart:io';

import 'package:dart_ml/tensor.dart' as t;

void main(List<String> arguments) {
  var input = t.Tensor([1, 2, 3, 2.5]);
  var weights = t.Tensor([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
  ]);
  var bias = t.Tensor([2, 3, 0.5]);

  var output = t.dot(weights, input) + bias;
  print(output.data); // 4.8 1.21 2.385
}
