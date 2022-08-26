class TensorTypeException implements Exception {
  final Type type, subtype;
  final String message;

  TensorTypeException(this.type, this.subtype, [this.message = '']);

  @override
  String toString() {
    return 'FormatException :\n A value of type $type is not subtype of $subtype';
  }
}
