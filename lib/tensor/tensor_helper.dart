class TensorHelper {
  static List<int> getShape(List matrix) {
    final temp = <int>[];
    dynamic current = matrix;
    while (current is List) {
      temp.add(current.length);
      current = current.first;
    }
    return temp;
  }

  static int getSize(List<int> shape) {
    var size = 1;
    for (var dim in shape) {
      size *= dim;
    }
    return size;
  }

  static List<int> getStride(List<int> shape) {
    var stride = List<int>.filled(shape.length, 0);
    var strideIndex = 1;
    for (var i = shape.length - 1; i >= 0; i--) {
      stride[i] = strideIndex;
      strideIndex *= shape[i];
    }
    return stride;
  }

  static List<int> defauldIndicesTable(int size) {
    var indices = <int>[];
    for (var i = 0; i < size; i++) {
      indices.add(i);
    }
    return indices;
  }

  static List<T> _getFlatList<T>(List matrix, List<int> shape) {
    var result = <T>[];
    if (shape.length == 1) {
      for (var i = 0; i < shape.last; i++) {
        result.add(matrix[i]);
      }
    } else if (shape.length == 2) {
      for (var i = 0; i < shape.first; i++) {
        for (var j = 0; j < shape.last; j++) {
          result.add(matrix[i][j]);
        }
      }
    } else {
      var dim = shape.removeAt(0);
      for (var i = 0; i < dim; i++) {
        result.addAll(_getFlatList(matrix[i], shape));
      }
      shape.insert(0, dim);
    }
    return result;
  }

  static List<T> flatListMatrix<T extends num>(
      {List matrix, List<int> shape, List<int> stride, int size}) {
    var flat = _getFlatList<T>(matrix, shape);

    var finalList = List<T>.filled(size, num.parse('0'));
    var currentDim = 0;
    var element = 0;
    if (shape.length > 2) {
      for (var i = 0; i < size; i++) {
        if (currentDim == shape.first) {
          currentDim = 0;
          element += 1;
        }

        var currentIndex = currentDim * stride.first + element;
        finalList[i] = flat[currentIndex];
        currentDim += 1;
      }
      return finalList;
    }

    return flat;
  }
}
