import 'package:dart_ml/tensor.dart' as t;
import 'package:dart_ml/neural_network.dart' as nn;
import 'dart:math' as math;

void main(List<String> arguments) {
  var x = t.Tensor.arange(24);
  x.reshape([2, 3, 4]);
  var arg = t.argmax(
    x,
    axis: 1,
  );
  print(arg.data);
}
