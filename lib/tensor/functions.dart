import 'package:dart_ml/tensor/tensor.dart';
import 'package:dart_ml/tensor/tensor_helper.dart';

Tensor dot<T extends num>(dynamic op1, dynamic op2) {
  Tensor t1;
  Tensor t2;
  if (op1 is! Tensor) {
    t1 = Tensor(op1);
  } else {
    t1 = op1;
  }
  if (op2 is! Tensor) {
    t2 = Tensor(op2);
  } else {
    t2 = op2;
  }

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
