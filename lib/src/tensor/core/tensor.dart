import 'package:dart_ml/src/tensor/errors/broadcast_exception.dart';
import 'package:dart_ml/src/tensor/errors/dynamic_type.dart';

import 'package:dart_ml/src/tensor/errors/tensor_type_exception.dart';

import '../errors/operationError.dart';
part 'tensor_helper.dart';
part 'num_tensor.dart';
part 'bool_tensor.dart';
part 'string_tensor.dart';

class Tensor<T> {
  final List<T> _tensor;
  List<int> _shape, _strides;
  List<List<int>> _indicesTable;
  int _size;
  bool _isScalar() => shape.isEmpty;
  List<int> get shape => List.from(_shape, growable: false);
  List<int> get strides => _strides;
  int get rank => _shape.length;
  int get size => _size;

  dynamic get data {
    if (_isScalar()) {
      return _tensor.first;
    } else {
      var output = _TensorHelper.createFromShape(_shape);
      var indeces = _TensorHelper.createIndicesTable(output, _shape, []);
      for (var i = 0; i < _size; i++) {
        _TensorHelper.addAtIndex(output, indeces[i], getAt([i]));
      }
      return output;
    }
  }

  Tensor._(
    this._tensor,
    this._shape,
    this._size,
    this._strides,
    this._indicesTable,
  );

  factory Tensor(dynamic input) {
    if (!_TensorHelper.isSupporetdType(input)) {
      throw TensorTypeException(input.runtimeType, List);
    }

    if (T == dynamic) {
      throw DynamicTypeException();
    }
    if (input is! List) {
      input = [input];
    }

    var inputShape = _TensorHelper.getShape(input);
    var matrix = _TensorHelper.rowMajor<T>(input, inputShape);
    var inputSize = _TensorHelper.initSize(inputShape);
    var indicesTable = _TensorHelper.createIndicesTable(input, inputShape, []);
    var inputStrides = _TensorHelper.initStride(inputShape);
    return Tensor<T>._(
      matrix,
      inputShape,
      inputSize,
      inputStrides,
      indicesTable,
    );
  }

  Tensor<T> copy() {
    return Tensor<T>._(
      List<T>.from(_tensor),
      List<int>.from(_shape),
      _size,
      List<int>.from(_strides),
      List<List<int>>.from(_indicesTable),
    );
  }

  Tensor<T> reshape(List<int> newShape) {
    var newSize = _TensorHelper.initSize(newShape);
    if (newSize != _size) {
      throw Exception('');
    }
    _shape = List.from(newShape);
    var _matrix = _TensorHelper.createFromShape(_shape);

    _strides = _TensorHelper.initStride(newShape);
    _indicesTable = _TensorHelper.createIndicesTable(_matrix, _shape, []);
    return this;
  }

  Tensor<E> getSubTensorAt<E extends T>(List other) {
    var output = [];
    return Tensor<E>(output);
  }

  T getAt(List<int> indcies) {
    List<int> location;
    if (indcies.length == 1) {
      location = _indicesTable[indcies.first];
    } else if (indcies.length == rank) {
      location = indcies;
    } else {
      throw Exception('');
    }
    return _tensor[_TensorHelper.getDataIndex(location, _strides)];
  }

  Tensor<num> operator +(Tensor other) {
    Tensor t;
    Tensor opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! num) {
      throw OperationError('+', T, other.runtimeType);
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

  Tensor<num> operator -(Tensor other) {
    Tensor t;
    Tensor opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! num) {
      throw OperationError('-', T, other.runtimeType);
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

  Tensor<num> operator *(Tensor other) {
    Tensor t;
    Tensor opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! num) {
      throw OperationError('*', T, other.runtimeType);
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

  Tensor<num> operator /(Tensor other) {
    Tensor t;
    Tensor opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! num) {
      throw OperationError('/', T, other.runtimeType);
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

  Tensor<bool> operator |(Tensor<T> other) {
    Tensor<T> t;
    Tensor<T> opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! bool) {
      throw OperationError('|', T, other.runtimeType);
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

  Tensor<bool> operator &(Tensor<T> other) {
    Tensor<T> t;
    Tensor<T> opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is bool) {
      throw OperationError('&', T, other.runtimeType);
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

  Tensor<bool> operator ^(Tensor<T> other) {
    Tensor<T> t;
    Tensor<T> opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! bool) {
      throw OperationError('^', T, other.runtimeType);
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

  Tensor<bool> operator ~() {
    var output = <bool>[];

    if (T is! bool) {
      throw OperationError('~', T);
    }
    for (var i = 0; i < _size; i++) {
      output.add(_indicesTable[i] as bool);
    }

    return Tensor<bool>(output)..reshape(_shape);
  }

  Tensor<bool> operator >(Tensor other) {
    Tensor t;
    Tensor opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! num) {
      throw OperationError('>', T, other.runtimeType);
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

  Tensor<bool> operator >=(Tensor other) {
    Tensor t;
    Tensor opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! num) {
      throw OperationError('>=', T, other.runtimeType);
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

  Tensor<bool> operator <(Tensor other) {
    Tensor t;
    Tensor opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw BroadcastException(opt.shape, t.shape);
    }
    if (other is! num) {
      throw OperationError('<', T, other.runtimeType);
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

  Tensor<bool> operator <=(Tensor other) {
    Tensor t;
    Tensor opt;

    var secondIsBigger = other.size > _size;

    if (secondIsBigger) {
      opt = other.copy();
      t = copy();
    } else {
      opt = copy();
      t = other.copy();
    }
    var broadcast = _TensorHelper.isBroadcastable(opt.shape, t.shape);

    if (broadcast == null) {
      throw Exception('unbroadcastable shapes');
    }
    if (other is! num) {
      throw OperationError('<=', T, other.runtimeType);
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

  @override
  bool operator ==(dynamic other) {
    Tensor<T> t;

    if (other is! Tensor) {
      t = Tensor<T>(other);
    }
    t = other;
    if (_TensorHelper.shapeEquality(t.shape, _shape)) {
      for (var i = 0; i < size; i++) {
        if (t._tensor[i] != _tensor[i]) {
          return false;
        }
      }

      return true;
    }
    return false;
  }

  Tensor<T> transpose({int? axis}) {
    if (rank == 1) {
      return copy();
    } else {
      var start = List<int>.filled(rank, 0);
      var newShape = List<int>.from(_shape.reversed);
      List<T> output = [];
      for (var i = 0; i < _size; i++) {
        output.add(getAt(List<int>.from(start.reversed)));
        _TensorHelper.addToShape(start, newShape);
      }
      return Tensor<T>(output)..reshape(newShape);
    }
  }

  @override
  Type get runtimeType => T;
}
