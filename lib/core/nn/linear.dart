import 'package:dart_ml/dl.dart';
import 'package:dtensor/src/core/dtensor.dart';

class Linear extends Neural {
  final DTensor<num> _w;
  final DTensor<num>? _c;

  Linear._(this._w, this._c);
  factory Linear(int inputFeatures, int outputFeatures, {bool bias = true}) {
    //TODO:change it to be random
    final DTensor<num> weight = DTensor.ones([inputFeatures, outputFeatures]);
    final DTensor<num> b = DTensor.ones([1, outputFeatures]);

    return Linear._(weight, bias ? b : null);
  }
  @override
  DTensor<num> forward(DTensor<num> input) {
    DTensor<num> val = input.dot(_w);
    if (_c != null) {
      val += _c!;
    }
    return val;
  }
}
