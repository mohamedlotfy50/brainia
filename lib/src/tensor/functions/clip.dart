// import 'package:dart_ml/src/tensor/core/tensor.dart';

// Tensor clip(dynamic input, num min, num max) {
//   var t1 = TensorHelper.toTensor(input);

//   var output = <num>[];
//   for (var i = 0; i < t1.size; i++) {
//     var t1Val = t1.getAt([i]);
//     num val;
//     if (min > max) {
//       val = max;
//     } else if (t1Val > max) {
//       val = max;
//     } else if (t1Val < min) {
//       val = min;
//     } else {
//       val = t1Val;
//     }
//     output.add(val);
//   }
//   var tensor = Tensor(output);
//   if (t1.shape.isNotEmpty) {
//     tensor.reshape(t1.shape);
//   }

//   return tensor;
// }
