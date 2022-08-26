class BroadcastException implements Exception {
  final List<int> enteredShape, shape;

  BroadcastException(this.enteredShape, this.shape);

  @override
  String toString() {
    return 'Can not broadcast shape $enteredShape to shape $shape';
  }
}
