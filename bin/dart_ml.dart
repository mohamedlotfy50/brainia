import 'package:dart_ml/src/tensor/core/tensor.dart';
import 'package:dart_ml/src/tensor/functions/dotproduct.dart';
import 'package:dart_ml/src/tensor/random/rand.dart';

void main(List<String> arguments) {
  Tensor t = Tensor<int>([1, 2, 3]);
  Tensor t2 = Tensor<int>([
    [1, 2, 3],
    [4, 5, 6]
  ]);
  print(t2.getAtT([0, 2]).data);
}
