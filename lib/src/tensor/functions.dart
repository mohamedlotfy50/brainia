import 'package:dart_ml/src/tensor/tensor.dart';
import 'package:dart_ml/src/tensor/tensor_helper.dart';
import 'dart:math' as math;

Tensor dot<T extends num>(dynamic op1, dynamic op2) {
  var t1 = TensorHelper.toTensor(op1), t2 = TensorHelper.toTensor(op2);

  if (t1.rank == 1 && t2.rank == 1) {
    if (TensorHelper.shapeEquality(t1.shape, t2.shape)) {
      return TensorHelper.vectorProduct(t1, t2);
    } else {
      throw Exception('dim execption');
    }
  } else if (t1.rank == 2 && t2.rank == 2) {
    if (t1.shape.last == t2.shape.first) {
      return TensorHelper.matrixMultiplication(t1, t2);
    } else {
      throw Exception('matrix dim execption');
    }
  } else if (t1.rank == 0 || t2.rank == 0) {
    return t1 * t2;
  } else if (t1.rank >= 2 && t2.rank == 1) {
    if (t1.shape.last == t2.shape.first) {
      return TensorHelper.muliplyOnAxis(t1, t2);
    } else {
      throw Exception('matrix dim execption');
    }
  } else if (t1.rank >= 2 || t2.rank >= 2) {
    throw Exception('un implemented method');

    // it is a sum product over the last axis of a and the second-to-last axis of b:
    // dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
  } else {
    throw Exception('unsuppored operation for those types');
  }
}

Tensor<T> maximum<T extends num>(dynamic input1, dynamic input2) {
  var t1 = TensorHelper.toTensor<T>(input1),
      t2 = TensorHelper.toTensor<T>(input2);

  if (t2.size > t1.size) {
    var temp = t1;
    t1 = t2;
    t2 = temp;
  }
  var output = <num>[];
  var broadcast = TensorHelper.isBroadcastable(t1.shape, t2.shape);
  if (broadcast != null) {
    for (var i = 0; i < t1.size; i++) {
      var t1Val = t1.getIndiceFromTable(i);
      var t2Val = t2.getIndiceFromTable(i % t2.size);
      if (t1Val > t2Val) {
        output.add(t1Val);
      } else {
        output.add(t2Val);
      }
    }

    return Tensor(output)..reshape(t1.shape);
  } else {
    throw Exception('not prodcastable');
  }
}

Tensor log(dynamic input) {
  var t1 = TensorHelper.toTensor(input);

  var output = <num>[];

  for (var i = 0; i < t1.size; i++) {
    var t1Val = t1.getIndiceFromTable(i);
    output.add(math.log(t1Val));
  }
  return Tensor(output)..reshape(t1.shape);
}

Tensor exp(dynamic input) {
  var t1 = TensorHelper.toTensor(input);

  var output = <num>[];
  for (var i = 0; i < t1.size; i++) {
    var t1Val = t1.getIndiceFromTable(i);
    output.add(math.pow(math.e, t1Val));
  }
  return Tensor(output)..reshape(t1.shape);
}

Tensor max(dynamic input, {int axis, bool keepDims = false}) {
  var t1 = TensorHelper.toTensor(input);

  if (axis == null) {
    num max = 0;
    for (var i = 0; i < t1.size; i++) {
      var val = t1.getIndiceFromTable(i);
      if (val > max) {
        max = val;
      }
    }
    return Tensor(max);
  } else if (axis <= t1.rank) {
    var newShape = List<int>.from(t1.shape);
    var breakingPoint = newShape.removeAt(axis);
    var currentShape = List<int>.filled(newShape.length, 0);
    var size = TensorHelper.initSize(newShape);
    var output = <num>[];

    for (var t = 0; t < size; t++) {
      num max = 0;
      for (var i = 0; i < breakingPoint; i++) {
        var location = List<int>.from(currentShape, growable: true);
        location.insert(axis, i);
        var val = t1.getElemetAt(location);
        if (val > max) {
          max = val;
        }
      }
      output.add(max);
      TensorHelper.addToShape(currentShape, newShape);
    }
    if (keepDims == true) {
      newShape.insert(axis, 1);
    }
    return Tensor(output)..reshape(newShape);
  } else {
    throw Exception('dim error');
  }
}

Tensor sum(dynamic input, {int axis, bool keepDims = false}) {
  var t1 = TensorHelper.toTensor(input);

  if (axis == null) {
    num total = 0;
    for (var i = 0; i < t1.size; i++) {
      total += t1.getIndiceFromTable(i);
    }
    return Tensor(total);
  } else if (axis <= t1.rank) {
    var newShape = List<int>.from(t1.shape);
    var breakingPoint = newShape.removeAt(axis);
    var currentShape = List<int>.filled(newShape.length, 0);
    var size = TensorHelper.initSize(newShape);
    var output = <num>[];

    for (var t = 0; t < size; t++) {
      num total = 0;
      for (var i = 0; i < breakingPoint; i++) {
        var location = List<int>.from(currentShape, growable: true);
        location.insert(axis, i);
        total += t1.getElemetAt(location);
      }
      output.add(total);
      TensorHelper.addToShape(currentShape, newShape);
    }
    if (keepDims == true) {
      newShape.insert(axis, 1);
    }
    return Tensor(output)..reshape(newShape);
  } else {
    throw Exception('dim error');
  }
}
