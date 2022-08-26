class OperationError implements Exception {
  final String operation;
  final Type type1;
  final Type? type2;

  OperationError(this.operation, this.type1, [this.type2]);
  @override
  String toString() {
    return 'operation ${operation} is not supported for ${type2 == null ? '$type1' : 'types $type1 and $type2'}';
  }
}
