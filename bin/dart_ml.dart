import 'package:dart_ml/src/activation/relu.dart';
import 'package:dart_ml/tensor.dart' as t;
import 'package:dart_ml/neural_network.dart' as nn;

void main(List<String> arguments) {
  var s = t.Tensor([-1, 2, 3, -6]);
  var a = ReLU();
  print(a.forward(s).data);
}
