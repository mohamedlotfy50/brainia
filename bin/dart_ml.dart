import 'package:dart_ml/tensor/tensor.dart';
import 'package:dart_ml/tensor/tensor_helper.dart';

void main(List<String> arguments) {
  var v = [1, 2, 3, 4];
  var v2 = [
    [1, 2, 3],
    [4, 5, 6]
  ];
  var v3 = [
    [
      [1, 2],
      [3, 4]
    ],
    [
      [5, 6],
      [7, 8]
    ]
  ];
  var t = Tensor.zeros([2, 3]);
  var t1 = Tensor(v);
  var t2 = Tensor<int>(v2);
  var t3 = Tensor<int>(v3);
  var t4 = Tensor(5);

  print(t.data);
  // print((t1 + 3).data);
  // print((t2 + 5).data);
  // print(t3.data);
  // print(t4.data);
}
