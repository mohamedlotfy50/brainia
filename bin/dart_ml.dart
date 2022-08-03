import 'package:dart_ml/tensor.dart' as t;
import 'package:dart_ml/neural_network.dart' as nn;

void main(List<String> arguments) {
  var s = t.Tensor([
    [4.8, 1.2, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]
  ]);

  var dens1 = nn.DenseLayer(inputLength: 3, units: 16);
  var activation1 = nn.ReLU();
  var dens2 = nn.DenseLayer(inputLength: 16, units: 2);
  var activation2 = nn.Softmax();

  var output = dens1.forward(s);
  output = activation1.forward(output);
  output = dens2.forward(output);
  output = activation2.forward(output);
  print(output.data);
}
