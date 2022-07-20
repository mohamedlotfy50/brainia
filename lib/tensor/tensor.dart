import 'package:dart_ml/tensor/tensor_helper.dart';
import 'dart:math' as math;

class Tensor<T extends num> {
  final List<T> _tensor;
  List<int> _shape, _strides;
  List<List<int>> _indicesTable;
  int _size;
  bool _isScalar() => shape.isEmpty;

  dynamic get data {
    if (_isScalar()) {
      return _tensor.first;
    } else {
      var output = TensorHelper.createFromShape(_shape);

      for (var i = 0; i < _size; i++) {
        var dataIndex = TensorHelper.dataIndex(_indicesTable[i], strides);
        TensorHelper.addAtIndex(output, _indicesTable[i], _tensor[dataIndex]);
      }
      return output;
    }
  }

  List<int> get shape => List.from(_shape, growable: false);
  List<int> get strides => _strides;
  int get rank => _shape.length;
  int get size => _size;

  Tensor._(
    this._tensor,
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
    var dataStrides = <int>[];

    if (data is num) {
      matrix.add(data as T);
      indicesTable.add([0]);
      dataStrides.add(1);
      dataSize = 1;
    } else {
      dataShape = TensorHelper.getShape(data);
      dataSize = TensorHelper.initSize(dataShape);

      matrix = TensorHelper.rowMajor<T>(data, dataShape);
      indicesTable = TensorHelper.createIndicesTable(data, dataShape, []);
      dataStrides = TensorHelper.initStride(dataShape);
    }

    return Tensor._(
      matrix,
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
      shape,
      dataSize,
      dataStrides,
      indicesTable,
    );
  }
  Tensor<T> copy() {
    return Tensor._(
      List<T>.from(_tensor),
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
      shape,
      dataSize,
      dataStrides,
      indicesTable,
    );
  }
  void reshape(List<int> newShape) {
    var newSize = TensorHelper.initSize(newShape);
    if (newSize != _size) {
      throw Exception('exception');
    } else {
      var newData = TensorHelper.createFromShape(newShape);
      _shape = newShape;

      _strides = TensorHelper.initStride(newShape);
      _indicesTable = TensorHelper.createIndicesTable(newData, newShape, []);
    }
  }

  Tensor<T> operator +(Tensor other) {
    Tensor<T> t;
    Tensor opt;

    if (other.size > _size) {
      opt = other.copy();
      t = this;
    } else {
      opt = copy();
      t = other;
    }
    print(TensorHelper.isBroadcastable(t.shape, opt.shape));
    if (TensorHelper.isBroadcastable(t.shape, opt.shape)) {
      for (var i = 0; i < opt.size; i++) {
        var index = TensorHelper.dataIndex(opt._indicesTable[i], opt._strides);
        opt._tensor[index] = opt._tensor[index] +
            t._tensor[TensorHelper.dataIndex(
                t._indicesTable[i % t.strides.first], t._strides)];
      }
      return opt;
    } else {
      throw Exception('unbroadcastable shapes');
    }
  }

  Tensor<T> operator -(dynamic other) {
    Tensor<T> t;
    Tensor opt;

    if (other.size > _size) {
      opt = other.copy();
      t = this;
    } else {
      opt = copy();
      t = other;
    }
    print(TensorHelper.isBroadcastable(t.shape, opt.shape));
    if (TensorHelper.isBroadcastable(t.shape, opt.shape)) {
      for (var i = 0; i < opt.size; i++) {
        var index = TensorHelper.dataIndex(opt._indicesTable[i], opt._strides);
        opt._tensor[index] = opt._tensor[index] -
            t._tensor[TensorHelper.dataIndex(
                t._indicesTable[i % t.strides.first], t._strides)];
      }
      return opt;
    } else {
      throw Exception('unbroadcastable shapes');
    }
  }

  @override
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

  Tensor add(dynamic other) {
    return this + other;
  }

  Tensor sub(dynamic other) {
    return this + other;
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
