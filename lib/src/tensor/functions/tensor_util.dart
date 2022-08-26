import 'package:dart_ml/src/tensor/core/tensor.dart';

class TensorHelper {
  static Tensor<T> toTensor<T extends num>(dynamic input) {
    late Tensor<T> t1;
    if (input is! Tensor<T>) {
      t1 = Tensor<T>(input);
    } else {
      t1 = input;
    }
    return t1;
  }

  static bool shapeEquality(List<int> shape1, List<int> shape2) {
    if (shape1.length == shape2.length) {
      for (var i = 0; i < shape1.length; i++) {
        if (shape1[i] != shape2[i]) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  static Tensor<V> vectorProduct<V extends Object>(Tensor t1, Tensor t2) {
    num total = 0;
    for (var i = 0; i < t1.size; i++) {
      total += t1.getAt([i]) * t2.getAt([i]);
    }
    return Tensor(total);
  }

  static Tensor matrixMultiplication(Tensor t1, Tensor t2) {
    var output = [];

    var newShape = [t1.shape.first, t2.shape.last];
    var row = 0;
    var column = 0;
    while (row < t1.shape.first) {
      num result = 0;

      for (var j = 0; j < t1.shape.last; j++) {
        var product = t1.getAt([j + row * t1.shape.last]) *
            t2.getAt([column + j * t2.shape.last]);

        result += product;
      }
      column += 1;

      if (column == t2.shape.last) {
        column = 0;
        row += 1;
      }

      output.add(result);
    }
    return Tensor(output)..reshape(newShape);
  }

  static Tensor<T> muliplyOnAxis<T extends num>(
      Tensor<num> t1, Tensor<num> t2) {
    var output = [];
    var newShape = List<int>.from(t1.shape);
    newShape.removeLast();

    var start = 0;

    while (start < t1.size) {
      num result = 0;
      for (var j = 0; j < t2.size; j++) {
        result += t1.getAt([start]) * t2.getAt([j]);
        start += 1;
      }
      output.add(result);
    }

    return Tensor(output)..reshape(newShape);
  }

  static Tensor<T> ndMultiplication<T extends num>() {
    var output = [];

    return Tensor(output)..reshape([]);
  }
}
