import 'package:dart_ml/tensor.dart' as t;

void main(List<String> arguments) {
  var input = t.Tensor([
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
  ]);
  var weights = t.Tensor([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
  ]);
  var weights2 = t.Tensor([
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
  ]);
  var bias = t.Tensor([2, 3, 0.5]);
  var bias2 = t.Tensor([1, 2, -0.5]);
  var output = t.dot(input, weights.transpose()) + bias;
  var output2 = t.dot(output, weights2) + bias2;
  print(output2.data);
}
