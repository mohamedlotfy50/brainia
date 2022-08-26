class DynamicTypeException implements Exception {
  @override
  String toString() {
    return 'Avoid dynamic type tensors:';
  }
}
