import 'package:dart_ml/tensor.dart' as t;

void main(List<String> arguments) {
  var t1 = t.Tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  print(t1.transpose().data);
  t1.reshape([2, 3]);
  print(t1.data);
}
