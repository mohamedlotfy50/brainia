part of '../core/tensor.dart';

Tensor<num> std(dynamic input,
    {int? axis, bool keepDims = false, List? where}) {
  var t1 = _TensorHelper.toTensor(input);

  if (axis == null) {
    num max = 0;
    for (var i = 0; i < t1.size; i++) {
      var val = t1.getAt([i]);
      if (val > max) {
        max = val;
      }
    }
    return Tensor(max);
  } else if (axis <= t1.rank) {
    var newShape = List<int>.from(t1.shape);
    var breakingPoint = newShape.removeAt(axis);
    var currentShape = List<int>.filled(newShape.length, 0);
    var size = _TensorHelper.initSize(newShape);
    var output = <num>[];

    for (var t = 0; t < size; t++) {
      num max = 0;
      for (var i = 0; i < breakingPoint; i++) {
        var location = List<int>.from(currentShape, growable: true);
        location.insert(axis, i);
        var val = t1.getAt(location);
        if (val > max) {
          max = val;
        }
      }
      output.add(max);
      _TensorHelper.addToShape(currentShape, newShape);
    }
    if (keepDims == true) {
      newShape.insert(axis, 1);
    }
    return Tensor(output)..reshape(newShape);
  } else {
    throw Exception('dim error');
  }
}
