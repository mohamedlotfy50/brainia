import '../lib/src/tensor/core/tensor.dart' as ts;

void main(List<String> arguments) {
  var t = ts.Tensor<int>([1, 2, 3]);
  var t2 = ts.Tensor<int>([
    [1, 2, 3],
    [4, 5, 6]
  ]);
  print(ts.sum(t).data);
}
