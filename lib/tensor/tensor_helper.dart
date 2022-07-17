class TensorHelper<T> {
  static int initSize(List<int> shape) {
    var size = 1;

    for (var dim in shape) {
      size *= dim;
    }

    return size;
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
    var last = indices.removeLast();
    for (var d in indices) {
      current = current[d];
    }
    current[last] = data;
    indices.add(last);
  }

  static List createFromShape<T>(List<int> datashape,
      {dynamic data, double Function() onGenerated}) {
    var last = datashape.removeLast();
    dynamic dim;
    if (data != null || onGenerated != null) {
      dim = List<T>.generate(last, (index) => data ?? onGenerated());
    } else {
      dim = List<T>.generate(last, (index) => null);
    }
    dynamic opList = [];

    for (var i = datashape.length - 1; i >= 0; i--) {
      for (var j = 0; j < datashape[i]; j++) {
        opList.add(List<T>.from(dim));
      }
      //TODO:fi error list is not int
      dim = List.from(opList);

      opList = [];
    }

    datashape.add(last);

    return dim;
  }

  static List<List<int>> createIndicesTable(
      List matrix, List<int> shape, List<int> indeces) {
    var output = <List<int>>[];

    if (matrix.first is! List) {
      var temp = <List<int>>[];
      for (var i = 0; i < matrix.length; i++) {
        var ind = List<int>.from(indeces);
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
}
