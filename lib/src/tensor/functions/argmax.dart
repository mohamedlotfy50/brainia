part of '../core/tensor.dart';

Tensor argmax(dynamic input, {int? axis}) {
  var t1 = _TensorHelper.toTensor(input);

  if (axis == null) {
    num max = 0;
    var index = 0;

    for (var i = 0; i < t1.size; i++) {
      var val = t1.getAt([i]);
      if (val > max) {
        max = val;
        index = i;
      }
    }
    return Tensor(index);
  } else if (axis <= t1.rank) {
    var newShape = List<int>.from(t1.shape);
    var breakingPoint = newShape.removeAt(axis);
    var currentShape = List<int>.filled(newShape.length, 0);
    var size = _TensorHelper.initSize(newShape);
    var output = <num>[];

    for (var t = 0; t < size; t++) {
      num max = 0;
      var index = 0;
      for (var i = 0; i < breakingPoint; i++) {
        var location = List<int>.from(currentShape, growable: true);
        location.insert(axis, i);
        var val = t1.getAt(location);
        if (val > max) {
          max = val;

          index = i;
        }
      }
      output.add(index);
      _TensorHelper.addToShape(currentShape, newShape);
    }

    return Tensor(output)..reshape(newShape);
  } else {
    throw Exception('dim error');
  }
}
