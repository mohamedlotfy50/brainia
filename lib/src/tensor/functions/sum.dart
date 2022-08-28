part of '../core/tensor.dart';

Tensor<T> sum<T extends num>(dynamic input,
    {int? axis, bool keepDims = false}) {
  var t1 = _TensorHelper.toTensor(input);

  if (axis == null) {
    num total = 0;
    for (var i = 0; i < t1.size; i++) {
      total += t1.getAt([i]);
    }
    return Tensor<T>(total);
  } else if (axis <= t1.rank) {
    var newShape = List<int>.from(t1.shape);
    var breakingPoint = newShape.removeAt(axis);
    var currentShape = List<int>.filled(newShape.length, 0);
    var size = _TensorHelper.initSize(newShape);
    var output = <num>[];

    for (var t = 0; t < size; t++) {
      num total = 0;
      for (var i = 0; i < breakingPoint; i++) {
        var location = List<int>.from(currentShape, growable: true);
        location.insert(axis, i);
        total += t1.getAt(location);
      }
      output.add(total);
      _TensorHelper.addToShape(currentShape, newShape);
    }
    if (keepDims == true) {
      newShape.insert(axis, 1);
    }
    return Tensor<T>(output)..reshape(newShape);
  } else {
    throw Exception('dim error');
  }
}
