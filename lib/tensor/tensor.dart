import 'dart:math';

import 'package:dart_ml/tensor/tensor_helper.dart';

class Tensor<T extends num> {
  List<T> _rowMatrix;
  final List<int> shape;
  final int rank;
  final int size;
  final List<int> stride;
  final List<int> indicesTable;

  Tensor._(this._rowMatrix,
      {this.shape, this.rank, this.size, this.stride, this.indicesTable});
  factory Tensor(List matrix) {
    var shape = TensorHelper.getShape(matrix);
    var rank = shape.length;
    var size = TensorHelper.getSize(shape);
    var stride = TensorHelper.getStride(shape);
    var indicesTable = TensorHelper.defauldIndicesTable(size);
    var rowMatrix = TensorHelper.flatListMatrix<T>(
      matrix: matrix,
      shape: shape,
      size: size,
      stride: stride,
    );

    return Tensor._(
      rowMatrix,
      indicesTable: indicesTable,
      rank: rank,
      shape: shape,
      size: size,
      stride: stride,
    );
  }

  Tensor view() {
    return this;
  }

  void operator +(dynamic other) {
    if (other is num) {
      for (var i = 0; i < size; i++) {
        _rowMatrix[i] += other;
      }
    } else if (other is List || other is Tensor) {
      Tensor op;
      if (other is List) {
        op = Tensor(other);
      } else {
        op = other;
      }
      if (size != op.size) {
        throw Exception('invalid shapes');
      }
      for (var i = 0; i < size; i++) {
        _rowMatrix[i] += op._rowMatrix[i];
      }
    } else {
      throw Exception('invalid type');
    }
  }

  void operator -(dynamic other) {
    if (other is num) {
      for (var i = 0; i < size; i++) {
        _rowMatrix[i] -= other;
      }
    } else if (other is List || other is Tensor) {
      Tensor op;
      if (other is List) {
        op = Tensor(other);
      } else {
        op = other;
      }
      if (size != op.size) {
        throw Exception('invalid shapes');
      }
      for (var i = 0; i < size; i++) {
        _rowMatrix[i] -= op._rowMatrix[i];
      }
    } else {
      throw Exception('invalid type');
    }
  }
}
