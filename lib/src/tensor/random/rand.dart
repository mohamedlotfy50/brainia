import 'dart:math';

class TensorRandom {
  late final Random _random;
  static TensorRandom? _instance;
  TensorRandom._([int? seed]) {
    if (seed == null) {
      _random = Random.secure();
    } else {
      _random = Random(seed);
    }
  }
  factory TensorRandom() {
    _instance ??= TensorRandom._();
    return _instance!;
  }

  factory TensorRandom.seed(int seed) {
    _instance ??= TensorRandom._(seed);
    return _instance!;
  }

  int nextInt(int max) {
    return _random.nextInt(max);
  }

  double nextDouble() {
    return _random.nextDouble();
  }

  double nrand() {
    return _random.nextBool() ? 1 : -1 * _random.nextDouble();
  }
}
