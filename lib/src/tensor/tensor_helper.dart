import 'package:dart_ml/src/tensor/tensor.dart';

class TensorHelper<T> {
  static int initSize(List<int> shape) {
    var size = 1;

    for (var dim in shape) {
      size *= dim;
    }

    return size;
  }

  static List<int> shapeMode(List<int> index, List<int> shape) {
    var currentIndex = List<int>.from(index);
    for (var i = 0; i < shape.length; i++) {
      currentIndex[i] %= shape[i];
    }
    return currentIndex;
  }

  static bool isIndexExist(List<List<int>> indices, List<int> index) {
    for (var i = 0; i < indices.length; i++) {
      var found = true;
      for (var j = 0; j < indices[i].length; j++) {
        if (indices[i][j] != indices[i][j]) {
          found = false;
          break;
        }
      }
      if (found) {
        return found;
      }
    }
    return false;
  }

  static int dataIndex<T>(
    List<int> indices,
    List<int> stride,
  ) {
    var index = 0;
    for (var i = 0; i < indices.length; i++) {
      index += indices[i] * stride[i];
    }

    return index;
  }

  List<int> defaultIndicesTable(int size) {
    var temp = <int>[];
    for (var i = 0; i < size; i++) {
      temp.add(i);
    }
    return temp;
  }

  static List<int> initStride(List<int> shape) {
    var temp = List<int>.generate(shape.length, (index) => null);
    var currentStride = 1;
    for (var i = shape.length - 1; i >= 0; i--) {
      temp[i] = currentStride;
      currentStride *= shape[i];
    }

    return temp;
  }

  static List<int> getShape(List data) {
    var temp = <int>[];

    dynamic current = data;
    while (current is List) {
      temp.add(current.length);
      current = current.first;
    }
    return temp;
  }

  static List<T> rowMajor<T>(
    List data,
    List<int> shape,
  ) {
    var temp = <T>[];
    if (data.first is! List) {
      for (var n in data) {
        temp.add(n);
      }
    } else {
      var dim = shape.removeAt(0);
      for (var i = 0; i < dim; i++) {
        temp.addAll(rowMajor(data[i], shape));
      }
      shape.insert(0, dim);
    }
    return temp;
  }

  static void addAtIndex(List operationList, List<int> indices, dynamic data) {
    dynamic current = operationList;
    var last = indices.last;
    for (var i = 0; i < indices.length - 1; i++) {
      current = current[indices[i]];
    }
    current[last] = data;
  }

  static List createFromShape(List<int> datashape,
      {dynamic data, double Function() onGenerated}) {
    var finalList = [];

    if (datashape.length == 1) {
      for (var i = 0; i < datashape.first; i++) {
        dynamic val;
        if (data != null || onGenerated != null) {
          val = data ?? onGenerated();
        }
        finalList.add(val);
      }
    } else {
      var first = datashape.removeAt(0);
      for (var i = 0; i < first; i++) {
        finalList.add(
          createFromShape(
            datashape,
            data: data,
            onGenerated: onGenerated,
          ),
        );
      }
      datashape.insert(0, first);
    }
    return finalList;
  }

  static List<List<int>> createIndicesTable(
      List matrix, List<int> shape, List<int> indeces) {
    var output = <List<int>>[];

    if (matrix.first is! List) {
      var temp = <List<int>>[];
      for (var i = 0; i < matrix.length; i++) {
        var ind = List<int>.from(indeces, growable: true);
        ind.add(i);

        temp.add(ind);
      }

      return temp;
    } else {
      var dim = shape.removeAt(0);
      for (var i = 0; i < dim; i++) {
        indeces.add(i);
        output.addAll(createIndicesTable(matrix[i], shape, indeces));
        indeces.removeLast();
      }
      shape.insert(0, dim);
    }

    return output;
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

  static int getDataIndex(List<int> indice, List<int> strides) {
    if (indice.length != strides.length) {
      throw Exception('unexpected erro');
    }
    var total = 0;
    for (var i = indice.length - 1; i >= 0; i--) {
      total += strides[i] * indice[i];
    }
    return total;
  }

  static List<int> isBroadcastable(List<int> shape1, List<int> shape2) {
    List<int> biggerShape, smallerShape;
    if (shape2.length > shape1.length) {
      biggerShape = List.from(shape2);
      smallerShape = List.from(shape1);
    } else {
      biggerShape = List.from(shape1);
      smallerShape = List.from(shape2);
    }

    var diff = biggerShape.length - smallerShape.length;
    for (var i = 0; i < diff; i++) {
      smallerShape.insert(0, 1);
    }

    for (var i = 0; i < biggerShape.length; i++) {
      var bi = biggerShape[i];
      var si = smallerShape[i];
      if (bi != si) {
        if (bi != 1 && si != 1) {
          return null;
        }
      }
    }

    return smallerShape;
  }

  static Tensor vectorProduct(Tensor t1, Tensor t2) {
    num total = 0;
    for (var i = 0; i < t1.size; i++) {
      total += t1.getElemetAt([i]) * t2.getElemetAt([i]);
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
        var product = t1.getIndiceFromTable(j + row * t1.shape.last) *
            t2.getIndiceFromTable(column + j * t2.shape.last);

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

  static Tensor<T> muliplyOnAxis<T extends num>(Tensor t1, Tensor t2) {
    var output = [];
    var newShape = List<int>.from(t1.shape);
    newShape.removeLast();

    var start = 0;

    while (start < t1.size) {
      num result = 0;
      for (var j = 0; j < t2.size; j++) {
        result += t1.getIndiceFromTable(start) * t2.getIndiceFromTable(j);
        start += 1;
      }
      output.add(result);
    }

    return Tensor(output)..reshape(newShape);
  }

  static Tensor<T> toTensor<T extends num>(dynamic input) {
    Tensor t1;
    if (input is! Tensor) {
      t1 = Tensor(input);
    } else {
      t1 = input;
    }
    return t1;
  }

  static void addToShape(List<int> addedShape, List<int> shape,
      {int amount = 1}) {
    for (var i = shape.length - 1; i >= 0; i--) {
      var s1 = addedShape[i];
      var s2 = shape[i];
      if (s1 == s2 - 1) {
        addedShape[i] = 0;
      } else {
        addedShape[i] = s1 + amount;
        break;
      }
    }
  }
}
