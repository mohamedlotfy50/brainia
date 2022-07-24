import 'package:dart_ml/src/tensor/tensor_helper.dart';
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
      var indeces = TensorHelper.createIndicesTable(output, _shape, []);
      for (var i = 0; i < _size; i++) {
        TensorHelper.addAtIndex(
            output, indeces[i], getElemetAt(_indicesTable[i]));
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
      _shape = newShape;
      _strides = TensorHelper.initStride(newShape);
    }
  }

  factory Tensor.arrange(int index) {
    var dataShape = [index];
    var data = List.generate(index, (i) => i);
    var dataSize = TensorHelper.initSize(dataShape);
    var matrix = TensorHelper.rowMajor<T>(data, dataShape);
    var dataStrides = TensorHelper.initStride(dataShape);
    var indicesTable = TensorHelper.createIndicesTable(data, dataShape, []);

    return Tensor._(
      matrix,
      dataShape,
      dataSize,
      dataStrides,
      indicesTable,
    );
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
    if (TensorHelper.isBroadcastable(t.shape, opt.shape)) {
      for (var i = 0; i < opt.size; i++) {
        var index = TensorHelper.dataIndex(opt._indicesTable[i], opt._strides);
        opt._tensor[index] =
            opt._tensor[index] + t.getIndiceFromTable(i % t.shape.last);
      }
      return opt;
    } else {
      throw Exception('unbroadcastable shapes');
    }
  }

  Tensor<T> operator -(Tensor other) {
    Tensor<T> t;
    Tensor opt;

    if (other.size > _size) {
      opt = other.copy();
      t = this;
    } else {
      opt = copy();
      t = other;
    }
    if (TensorHelper.isBroadcastable(t.shape, opt.shape)) {
      for (var i = 0; i < opt.size; i++) {
        var index = TensorHelper.dataIndex(opt._indicesTable[i], opt._strides);
        opt._tensor[index] =
            opt._tensor[index] - t.getIndiceFromTable(i % t.shape.last);
      }
      return opt;
    } else {
      throw Exception('unbroadcastable shapes');
    }
  }

  Tensor<T> operator *(Tensor other) {
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
        opt._tensor[index] =
            opt._tensor[index] * t.getIndiceFromTable(i % t.shape.last);
      }
      return opt;
    } else {
      throw Exception('unbroadcastable shapes');
    }
  }

  Tensor<T> operator /(Tensor other) {
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
        opt._tensor[index] =
            opt._tensor[index] / t.getIndiceFromTable(i % t.shape.last);
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

  T getElemetAt(List<int> index) {
    if (index.length == rank) {
      return _tensor[TensorHelper.getDataIndex(index, strides)];
    } else {
      throw Exception('high rank');
    }
  }

  T getIndiceFromTable(int i) {
    if (i < size) {
      return _tensor[TensorHelper.getDataIndex(_indicesTable[i], _strides)];
    } else {
      throw Exception('out of range');
    }
  }

  Tensor<T> transpose() {
    if (rank == 1) {
      return this;
    } else if (rank == 2) {
      var newIndicesTable = <List<int>>[];

      var currentIndex = 0;
      var currentdim = 0;

      for (var i = 0; i < size; i++) {
        newIndicesTable
            .add(_indicesTable[currentIndex + strides.first * currentdim]);
        currentdim += 1;
        if (currentdim == shape.first) {
          currentIndex += 1;
          currentdim = 0;
        }
      }
      _indicesTable = newIndicesTable;

      _shape = [shape.last, shape.first];
      return this;
    } else {
      // add nd transpose
    }
  }
}
