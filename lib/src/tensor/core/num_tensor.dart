part of 'tensor.dart';

class _NumTensor {
  static Tensor<bool> smallerOrEqual<T>(Tensor t1, Tensor t2) {
    Tensor t;
    Tensor opt;

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
      throw Exception('unbroadcastable shapes');
    }
    if (t2 is! num) {
      throw OperationError('<=', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <bool>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));
      var sum;
      if (secondIsBigger) {
        sum = val1 >= val2;
      } else {
        sum = val1 <= val2;
      }

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<bool>(output)..reshape(opt.shape);
  }

  static Tensor<bool> smaller<T>(Tensor t1, Tensor t2) {
    Tensor t;
    Tensor opt;

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
      throw Exception();
    }
    if (t2 is! num) {
      throw OperationError('<', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <bool>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));
      var sum;
      if (secondIsBigger) {
        sum = val1 > val2;
      } else {
        sum = val1 < val2;
      }

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<bool>(output)..reshape(opt.shape);
  }

  static Tensor<bool> greaterOrEqual<T>(Tensor t1, Tensor t2) {
    Tensor t;
    Tensor opt;

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
      throw Exception();
    }
    if (t2 is! num) {
      throw OperationError('>=', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <bool>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));
      var sum;
      if (secondIsBigger) {
        sum = val1 <= val2;
      } else {
        sum = val1 >= val2;
      }

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<bool>(output)..reshape(opt.shape);
  }

  static Tensor<bool> greaterThan<T>(Tensor t1, Tensor t2) {
    Tensor t;
    Tensor opt;

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
      throw Exception();
    }
    if (t2 is! num) {
      throw OperationError('>', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <bool>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));
      var sum;
      if (secondIsBigger) {
        sum = val1 < val2;
      } else {
        sum = val1 > val2;
      }

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<bool>(output)..reshape(opt.shape);
  }

  static Tensor<num> add<T>(Tensor t1, Tensor t2) {
    Tensor t;
    Tensor opt;

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
      throw Exception();
    }
    if (t2 is! num) {
      throw OperationError('+', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <num>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));

      var sum = val1 + val2;

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<num>(output)..reshape(opt.shape);
  }

  static Tensor<num> subtract<T>(Tensor t1, Tensor t2) {
    Tensor t;
    Tensor opt;

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
      throw Exception();
    }
    if (t2 is! num) {
      throw OperationError('-', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <num>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));
      var sum;

      if (secondIsBigger) {
        sum = val2 - val1;
      } else {
        sum = val1 - val2;
      }

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<num>(output)..reshape(opt.shape);
  }

  static Tensor<num> multiply<T>(Tensor t1, Tensor t2) {
    Tensor t;
    Tensor opt;

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
      throw Exception();
    }
    if (t2 is! num) {
      throw OperationError('*', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <num>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));

      var sum = val1 * val2;

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<num>(output)..reshape(opt.shape);
  }

  static Tensor<num> divid<T>(Tensor t1, Tensor t2) {
    Tensor t;
    Tensor opt;

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
      throw Exception();
    }
    if (t2 is! num) {
      throw OperationError('/', T, t2.runtimeType);
    }
    t.reshape(broadcast);
    var output = <num>[];
    var currentIndex = List<int>.filled(opt.shape.length, 0);
    for (var i = 0; i < opt.size; i++) {
      final val1 = opt.getAt(currentIndex);
      final val2 = t.getAt(_TensorHelper.shapeMode(currentIndex, t.shape));
      var sum;

      if (secondIsBigger) {
        sum = val2 / val1;
      } else {
        sum = val1 / val2;
      }

      output.add(sum);
      _TensorHelper.addToShape(currentIndex, opt.shape);
    }
    return Tensor<num>(output)..reshape(opt.shape);
  }
}
