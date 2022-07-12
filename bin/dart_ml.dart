import 'package:dart_ml/tensor/tensor.dart';

void main(List<String> arguments) {
  var n1 = 3, n2 = 5;
  var v1 = [1, 2, 3, 4];
  var t1 = Tensor<int>(v1);

  var shape1 = [4];
  var v2 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
  ];
  var shape2 = [2, 4];
  var t2 = Tensor<int>(v2);

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
  var t3 = Tensor(v3);

  var shape3 = [2, 2, 2];
  var v4 = [
    [
      [1, 2],
      [3, 4]
    ],
    [
      [4, 5],
      [6, 7]
    ],
    [
      [8, 9],
      [10, 11]
    ]
  ];
  var shape4 = [3, 2, 2];
  var t4 = Tensor(v4);
}
