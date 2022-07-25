import 'package:dart_ml/src/tensor/tensor.dart';
import 'package:dart_ml/tensor.dart' as t;

class DenseLayer {
  Tensor _weights, _biases, _outputs;

  Tensor get weights => _weights;
  Tensor get bias => _biases;
  Tensor get outputs => _outputs;
  DenseLayer._(
    this._weights,
    this._biases,
  );
  factory DenseLayer({
    int inputLength = 0,
    int units = 0,
  }) {
    return DenseLayer._(
        Tensor.randn([inputLength, units]), Tensor.zeros([1, units]));
  }

  Tensor forward(Tensor input) {
    _outputs = t.dot(input, _weights) + _biases;
    return _outputs;
  }
}
