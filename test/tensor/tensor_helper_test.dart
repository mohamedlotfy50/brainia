import 'package:dart_ml/tensor/tensor_helper.dart';
import 'package:test/test.dart';

void main() {
  test('Test if shape is broadcastable', () {
    expect(TensorHelper.isBroadcastable([2, 3], [2, 3]), true);
    expect(TensorHelper.isBroadcastable([2, 3], [2, 1]), true);
    expect(TensorHelper.isBroadcastable([2, 3], [1, 3]), true);
    expect(TensorHelper.isBroadcastable([2, 3], [3]), true);
    expect(
        TensorHelper.isBroadcastable(
            [1000, 256, 256, 256], [1000, 256, 256, 256]),
        true);
    expect(
        TensorHelper.isBroadcastable(
            [1000, 256, 256, 256], [1000, 1, 256, 256]),
        true);
    expect(
        TensorHelper.isBroadcastable([1000, 256, 256, 256], [1000, 1, 1, 256]),
        true);
    expect(
        TensorHelper.isBroadcastable([1000, 256, 256, 256], [1, 256, 256, 256]),
        true);
    expect(TensorHelper.isBroadcastable([1000, 256, 256, 256], [1, 1, 256, 1]),
        true);
    expect(TensorHelper.isBroadcastable([2, 3], [3, 2]), false);
    expect(TensorHelper.isBroadcastable([2, 3], [2]), false);
    expect(
        TensorHelper.isBroadcastable([1000, 256, 256, 256], [1000, 256, 256]),
        false);
  });
}
