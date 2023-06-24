import 'package:dart_ml/dl.dart';
import 'package:dtensor/src/core/dtensor.dart';

class Sequencial extends Neural {
  final List<Neural> layers;

  Sequencial(this.layers);
  @override
  DTensor<num> forward(DTensor<num> input) {
    DTensor<num> value = input;

    for (Neural layer in layers) {
      value = layer(value);
    }

    return value;
  }
}
