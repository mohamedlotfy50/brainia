import 'package:dart_ml/tensor.dart' as t;
import 'package:dart_ml/neural_network.dart' as nn;

void main(List<String> arguments) {
  var input = t.Tensor([
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
  ]);
  var layer1 = nn.DenseLayer(inputLength: 4, units: 5);
  var layer2 = nn.DenseLayer(inputLength: 5, units: 2);
  var out1 = layer1.forward(input);
  var out2 = layer2.forward(out1);
  print(out2.data);
}
