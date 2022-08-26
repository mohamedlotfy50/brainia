part of 'tensor.dart';

class _BoolTensor {
  static Tensor<bool> bitwise<T>(Tensor<T> t) {
    var output = <bool>[];

    if (T is! bool) {
      throw OperationError('~', T);
    }
    for (var i = 0; i < t._size; i++) {
      output.add(t._indicesTable[i] as bool);
    }

    return Tensor<bool>(output)..reshape(t._shape);
  }

  static Tensor<bool> xor<T>(Tensor<T> t1, Tensor<T> t2) {
    Tensor<T> t;
    Tensor<T> opt;

    var secondIsBigger = t2.size > t1._size;

    if (secondIsBigger) {
      opt = t2.copy();
      t = t1.copy();
    } else {
      opt = t1.copy();
      t = t2.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (t2 is! bool) {
      throw OperationError('^', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <bool>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));

      var sum = (val1 as bool) ^ (val2 as bool);

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<bool>(output)..reshape(opt.shape);
  }

  static Tensor<bool> and<T>(Tensor<T> t1, Tensor<T> t2) {
    Tensor<T> t;
    Tensor<T> opt;

    var secondIsBigger = t2.size > t1._size;

    if (secondIsBigger) {
      opt = t2.copy();
      t = t1.copy();
    } else {
      opt = t1.copy();
      t = t2.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (t2 is bool) {
      throw OperationError('&', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <bool>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));

      var sum = (val1 as bool) & (val2 as bool);

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<bool>(output)..reshape(opt.shape);
  }

  static Tensor<bool> or<T>(Tensor<T> t1, Tensor<T> t2) {
    Tensor<T> t;
    Tensor<T> opt;

    var secondIsBigger = t2.size > t1._size;

    if (secondIsBigger) {
      opt = t2.copy();
      t = t1.copy();
    } else {
      opt = t1.copy();
      t = t2.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (t2 is! bool) {
      throw OperationError('|', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <bool>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));

      var sum = (val1 as bool) | (val2 as bool);

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<bool>(output)..reshape(opt.shape);
  }
}
