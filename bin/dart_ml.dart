import 'package:dart_ml/tensor/tensor.dart';
import 'package:dart_ml/tensor/tensor_helper.dart';

void main(List<String> arguments) {
  var zeros = Tensor.zeros([2, 2]);
  var ones = Tensor.ones([2, 2, 2]);
  var random = Tensor.rand([4]);

  print(zeros.data);
  print(ones.data);
  print(random.data);
}
