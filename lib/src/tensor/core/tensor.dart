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

  Tensor<T> getAtT(List<int> index) {
    var _indexSize = index.length;
    if (_indexSize > rank) {
      throw Exception('');
    }
    var newShape = List<int>.from(_shape);
    var start = 0;
    var end = 0;
    for (var i = 0; i < _indexSize; i++) {}

    var output = Tensor<T>(_tensor.sublist(start, end == 0 ? start : end));
    if (end != 0) {
      output = output.reshape(newShape);
    }

    return output;
  }

  Tensor<num> operator +(Tensor other) {
    return _NumTensor.add<T>(this, other);
  }

  Tensor<num> operator -(Tensor other) {
    return _NumTensor.subtract<T>(this, other);
  }

  Tensor<num> operator *(Tensor other) {
    return _NumTensor.multiply<T>(this, other);
  }

  Tensor<num> operator /(Tensor other) {
    return _NumTensor.divid<T>(this, other);
  }

  Tensor<bool> operator |(Tensor<T> other) {
    return _BoolTensor.or<T>(this, other);
  }

  Tensor<bool> operator &(Tensor<T> other) {
    return _BoolTensor.and<T>(this, other);
  }

  Tensor<bool> operator ^(Tensor<T> other) {
    return _BoolTensor.xor<T>(this, other);
  }

  Tensor<bool> operator ~() {
    return _BoolTensor.bitwise<T>(this);
  }

  Tensor<bool> operator >(Tensor other) {
    return _NumTensor.greaterThan<T>(this, other);
  }

  Tensor<bool> operator >=(Tensor<bool> other) {
    return _NumTensor.greaterOrEqual<T>(this, other);
  }

  Tensor<bool> operator <(Tensor<bool> other) {
    return _NumTensor.smaller<T>(this, other);
  }

  Tensor<bool> operator <=(Tensor<bool> other) {
    return _NumTensor.smallerOrEqual<T>(this, other);
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
      var output = <T>[];
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
