import 'package:dart_ml/tensor/tensor_helper.dart';
import 'dart:math' as math;

class Tensor<T extends num> {
  final List<T> _tensor;
  final bool _isScalar;
  List<int> _shape, _strides;
  List<List<int>> _indicesTable;
  int _size;
  dynamic get data {
    if (_isScalar) {
      return _tensor.first;
    } else {
      var output = TensorHelper.createFromShape(_shape);

      for (var i = 0; i < _size; i++) {
        TensorHelper.addAtIndex(output, _indicesTable[i], _tensor[i]);
      }
      return output;
    }
  }

  List<int> get shape => _shape;
  List<int> get strides => _strides;
  int get rank => _shape.length;
  int get size => _size;

  Tensor._(
    this._tensor,
    this._isScalar,
    this._shape,
    this._size,
    this._strides,
    this._indicesTable,
  );

  factory Tensor(dynamic data) {
    var matrix = <T>[];
    var dataShape = <int>[];
    int dataSize;
    var indicesTable = <List<int>>[];

    if (data is num) {
      matrix.add(data as T);
    } else {
      dataShape = TensorHelper.getShape(data);
      dataSize = TensorHelper.initSize(dataShape);

      matrix = TensorHelper.rowMajor<T>(data, dataShape);
      indicesTable = TensorHelper.createIndicesTable(data, dataShape, []);
    }

    var dataStrides = TensorHelper.initStride(dataShape);

    return Tensor._(
      matrix,
      data is num,
      dataShape,
      dataSize,
      dataStrides,
      indicesTable,
    );
  }
  factory Tensor.zeros(List<int> shape) {
    var data = TensorHelper.createFromShape(shape, data: 0);
    var dataSize = TensorHelper.initSize(shape);
    var matrix = TensorHelper.rowMajor<T>(data, shape);
    var dataStrides = TensorHelper.initStride(shape);
    var indicesTable = TensorHelper.createIndicesTable(data, shape, []);

    return Tensor._(
      matrix,
      false,
      shape,
      dataSize,
      dataStrides,
      indicesTable,
    );
  }
  factory Tensor.ones(List<int> shape) {
    var data = TensorHelper.createFromShape(shape, data: 1);
    var dataSize = TensorHelper.initSize(shape);
    var matrix = TensorHelper.rowMajor<T>(data, shape);
    var dataStrides = TensorHelper.initStride(shape);
    var indicesTable = TensorHelper.createIndicesTable(data, shape, []);

    return Tensor._(
      matrix,
      false,
      shape,
      dataSize,
      dataStrides,
      indicesTable,
    );
  }
  Tensor<T> copy() {
    return Tensor._(
      List<T>.from(_tensor),
      _isScalar,
      List<int>.from(_shape),
      _size,
      List<int>.from(_strides),
      List<List<int>>.from(_indicesTable),
    );
  }

  factory Tensor.rand(List<int> shape) {
    var data = TensorHelper.createFromShape(shape,
        onGenerated: math.Random.secure().nextDouble);
    var dataSize = TensorHelper.initSize(shape);
    var matrix = TensorHelper.rowMajor<T>(data, shape);
    var dataStrides = TensorHelper.initStride(shape);
    var indicesTable = TensorHelper.createIndicesTable(data, shape, []);

    return Tensor._(
      matrix,
      false,
      shape,
      dataSize,
      dataStrides,
      indicesTable,
    );
  }
  void reshape(List<int> newShape) {
    var newSize = TensorHelper.initSize(newShape);
    if (newSize != _size || _isScalar) {
      throw Exception('exception');
    } else if (data is List) {
      var newEmptyShape = TensorHelper.createFromShape(newShape);
      _shape = newShape;

      _strides = TensorHelper.initStride(newShape);
      _indicesTable =
          TensorHelper.createIndicesTable(newEmptyShape, newShape, []);
    }
  }

  Tensor<T> operator +(dynamic other) {
    Tensor<T> t;
    var op = copy();

    if (other is! Tensor) {
      t = Tensor<T>(other);
    }
    t = other;
    if (op._isScalar || t._isScalar) {
      if (op._isScalar) {
        op = t.copy();
        t = copy();
      }

      for (var i = 0; i < op.size; i++) {
        op._tensor[i] += t.data;
      }
    } else if (TensorHelper.shapeEquality(t.shape, op.shape)) {
      for (var i = 0; i < op.size; i++) {
        op._tensor[i] += t._tensor[i];
      }
    } else {
      throw Exception('invalid operation');
    }

    return op;
  }

  Tensor<T> operator -(dynamic other) {
    Tensor<T> t;
    var op = copy();

    if (other is! Tensor) {
      t = Tensor<T>(other);
    }
    t = other;
    if (op._isScalar || t._isScalar) {
      if (op._isScalar) {
        op = t.copy();
        t = copy();
      }

      for (var i = 0; i < op.size; i++) {
        op._tensor[i] -= t.data;
      }
    } else if (TensorHelper.shapeEquality(t.shape, op.shape)) {
      for (var i = 0; i < op.size; i++) {
        op._tensor[i] -= t._tensor[i];
      }
    } else {
      throw Exception('invalid operation');
    }

    return op;
  }

  bool operator ==(dynamic other) {
    Tensor<T> t;

    if (other is! Tensor) {
      t = Tensor<T>(other);
    }
    t = other;
    if (TensorHelper.shapeEquality(t.shape, _shape)) {
      for (var i = 0; i < size; i++) {
        if (t._tensor[i] != _tensor[i]) {
          return false;
        }
      }

      return true;
    } else {
      return false;
    }
  }

  T max() {
    T max;
    for (var i = 0; i < size; i++) {
      if (max == null) {
        max = _tensor[i];
      } else if (_tensor[i] > max) {
        max = _tensor[i];
      }
    }
    return max;
  }

  T min() {
    T min;
    for (var i = 0; i < size; i++) {
      if (min == null) {
        min = _tensor[i];
      } else if (_tensor[i] < min) {
        min = _tensor[i];
      }
    }
    return min;
  }
}
